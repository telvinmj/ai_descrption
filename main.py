# backend/main.py

import os
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
import yaml
import json
from pathlib import Path
import tempfile

from backend.services.metadata_service import MetadataService
from backend.services.file_watcher_service import FileWatcherService

# Load environment variables
load_dotenv()

# Get dbt projects directory from environment (set either in run.py or .env)
dbt_projects_dir = os.environ.get("DBT_PROJECTS_DIR", "dbt_projects_2")

# Check if it's an absolute path and use it directly if so
if os.path.isabs(dbt_projects_dir):
    print(f"Using absolute path for dbt_projects_dir: {dbt_projects_dir}")
else:
    # If relative, keep as is - metadata_service will resolve it relative to base_dir
    print(f"Using relative path for dbt_projects_dir: {dbt_projects_dir}")

# Initialize FastAPI app
app = FastAPI(title="DBT Metadata Explorer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize metadata service with the specified projects directory
metadata_service = MetadataService(dbt_projects_dir=dbt_projects_dir)

# Initialize file watcher service with the same projects directory
file_watcher = FileWatcherService(
    dbt_projects_dir=metadata_service.dbt_projects_dir,
    refresh_callback=metadata_service.refresh,
    watch_interval=int(os.environ.get("WATCHER_POLL_INTERVAL", 30))  # Check interval in seconds
)

# Start file watcher on startup (disabled by default)
@app.on_event("startup")
async def startup_event():
    # Initialize the file watcher but don't start it automatically
    # The user can enable it manually through the UI
    print("Auto-refresh is OFF by default. Enable it from the UI if needed.")
    # file_watcher.start()
    # print("File watcher started for automatic metadata updates")

# Stop file watcher on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    file_watcher.stop()
    print("File watcher stopped")

# Define models for request bodies
class DescriptionUpdate(BaseModel):
    description: str

@app.get("/api/projects")
async def get_projects():
    """Get all dbt projects"""
    return metadata_service.get_projects()

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get a specific project by ID"""
    projects = metadata_service.get_projects()
    for project in projects:
        if project["id"] == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")

@app.get("/api/models")
async def get_models(project_id: str = None, search: str = None, tag: str = None, materialized: str = None):
    """Get all models, optionally filtered by project, search term, tag, or materialization type"""
    # Log search parameters for debugging
    print(f"API GET /api/models with params: project_id={project_id}, search='{search}', tag={tag}, materialized={materialized}")
    
    # Ensure search is properly handled
    if search:
        search = search.strip()
        print(f"Performing exact name match search for: '{search}'")
    
    # Get models from service with exact name matching
    models = metadata_service.get_models(project_id, search)
    
    # Apply additional filters
    if tag or materialized:
        filtered_models = []
        for model in models:
            # Filter by tag if specified
            if tag:
                model_tags = model.get("tags", [])
                if not model_tags or tag not in model_tags:
                    continue
                    
            # Filter by materialization type if specified
            if materialized:
                if model.get("materialized") != materialized:
                    continue
                    
            filtered_models.append(model)
        
        print(f"After tag/materialized filtering: {len(filtered_models)} models remaining")
        models = filtered_models
    
    # Add default values for missing fields
    defaults = {
        "description": "",
        "schema": "default",
        "materialized": "view",
        "file_path": "N/A",
        "columns": [],
        "sql": "",
        "tags": []
    }
    
    # Apply defaults to each model
    for model in models:
        for key, default_value in defaults.items():
            if key not in model or model[key] is None:
                model[key] = default_value
        
        # Make sure empty descriptions use AI descriptions if available
        if model.get("description") == "" and model.get("ai_description"):
            model["description"] = model["ai_description"]
        
        # Process columns to fill in empty descriptions with AI descriptions
        for column in model.get("columns", []):
            if column.get("description") == "" and column.get("ai_description"):
                column["description"] = column["ai_description"]
    
    print(f"API returning {len(models)} models")
    if search and len(models) > 0:
        print(f"Returned model names: {', '.join(m['name'] for m in models)}")
    
    return models

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model by ID"""
    model = metadata_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Make sure empty descriptions use AI descriptions if available
    if model.get("description") == "" and model.get("ai_description"):
        model["description"] = model["ai_description"]
    
    # Process columns to fill in empty descriptions with AI descriptions
    for column in model.get("columns", []):
        if column.get("description") == "" and column.get("ai_description"):
            column["description"] = column["ai_description"]
            
    return model

@app.get("/api/models/{model_id}/lineage")
async def get_model_lineage(model_id: str):
    """Get a model with its lineage information"""
    if model_id == "NaN" or not model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")
        
    model = metadata_service.get_model_with_lineage(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.get("/api/lineage")
async def get_lineage():
    """Get all lineage relationships"""
    return metadata_service.get_lineage()

@app.post("/api/models/{model_id}/description")
async def update_model_description(model_id: str, update: DescriptionUpdate):
    """Update a model's description"""
    success = metadata_service.update_description("model", model_id, update.description)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found or update failed")
    return {"status": "success", "message": "Description updated successfully"}

@app.post("/api/columns/{model_id}/{column_name}/description")
async def update_column_description(model_id: str, column_name: str, update: DescriptionUpdate):
    """Update a column's description"""
    entity_id = f"{model_id}:{column_name}"
    success = metadata_service.update_description("column", entity_id, update.description)
    if not success:
        raise HTTPException(status_code=404, detail="Column not found or update failed")
    return {"status": "success", "message": "Description updated successfully"}

@app.post("/api/refresh")
async def refresh_metadata():
    """Refresh metadata from dbt projects"""
    success = metadata_service.refresh()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to refresh metadata")
    return {"status": "success", "message": "Metadata refreshed successfully"}

@app.post("/api/models/{model_id}/refresh")
async def refresh_model_metadata(model_id: str):
    """Refresh AI descriptions for a specific model"""
    model = metadata_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Ensure AI descriptions are enabled
    if not metadata_service.use_ai_descriptions:
        print("Enabling AI descriptions for model refresh")
        metadata_service.use_ai_descriptions = True
        if not metadata_service.ai_service:
            from backend.services.ai_description_service import AIDescriptionService
            metadata_service.ai_service = AIDescriptionService()
    
    success = metadata_service.refresh_model_metadata(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to refresh model metadata")
    
    # Get the updated model data
    updated_model = metadata_service.get_model(model_id)
    if not updated_model:
        raise HTTPException(status_code=404, detail="Updated model not found")
        
    return updated_model

@app.get("/api/export/json")
async def export_metadata_json():
    """Export all metadata in JSON format"""
    # Get all projects and models
    projects = metadata_service.get_projects()
    models = metadata_service.get_models()
    lineage = metadata_service.get_lineage()
    
    # Create export structure
    export_data = {
        "metadata_version": "1.0",
        "projects": projects,
        "models": models,
        "lineage": lineage
    }
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_file.write(json.dumps(export_data, indent=2).encode('utf-8'))
        temp_path = temp_file.name
    
    # Return the file as a download
    return FileResponse(
        path=temp_path, 
        filename="dbt_metadata_export.json",
        media_type="application/json",
        background=lambda: os.unlink(temp_path)  # Delete the file after sending
    )

@app.get("/api/export/yaml")
async def export_metadata_yaml():
    """Export all metadata in YAML format"""
    # Get all projects and models
    projects = metadata_service.get_projects()
    models = metadata_service.get_models()
    lineage = metadata_service.get_lineage()
    
    # Create export structure
    export_data = {
        "metadata_version": "1.0",
        "projects": projects,
        "models": models,
        "lineage": lineage
    }
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        temp_file.write(yaml.dump(export_data, sort_keys=False).encode('utf-8'))
        temp_path = temp_file.name
    
    # Return the file as a download
    return FileResponse(
        path=temp_path, 
        filename="dbt_metadata_export.yaml",
        media_type="application/yaml",
        background=lambda: os.unlink(temp_path)  # Delete the file after sending
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/watcher/status")
async def get_watcher_status():
    """Get the current status of the file watcher"""
    return file_watcher.get_status()

@app.post("/api/watcher/toggle")
async def toggle_watcher(enable: bool = True):
    """Enable or disable the file watcher"""
    if enable:
        success = file_watcher.start()
        if success:
            return {"status": "success", "message": "File watcher started"}
        else:
            return {"status": "warning", "message": "File watcher already running"}
    else:
        success = file_watcher.stop()
        if success:
            return {"status": "success", "message": "File watcher stopped"}
        else:
            return {"status": "warning", "message": "File watcher already stopped"}

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)