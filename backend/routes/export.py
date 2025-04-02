from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

from backend.models.schema import MetadataExport
from backend.models.database import (
    Project as DBProject, 
    Model as DBModel, 
    ColumnModel as DBColumn,
    Lineage as DBLineage
)
from backend.services.database import get_db

router = APIRouter(
    prefix="/export",
    tags=["export"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=MetadataExport)
def export_metadata(db: Session = Depends(get_db)):
    """Export all metadata as a structured JSON document"""
    
    # Get all projects with eager loading of related data
    projects = db.query(DBProject).all()
    models = db.query(DBModel).all()
    columns = db.query(DBColumn).all()
    lineage = db.query(DBLineage).all()
    
    # Create serializable data structures
    projects_data = []
    for project in projects:
        projects_data.append({
            "id": project.id,
            "name": project.name,
            "path": project.path,
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None
        })
    
    models_data = []
    for model in models:
        models_data.append({
            "id": model.id,
            "name": model.name,
            "project_id": model.project_id,
            "project_name": model.project.name,
            "file_path": model.file_path,
            "schema": model.schema,
            "materialized": model.materialized,
            "description": model.description,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        })
    
    columns_data = []
    for column in columns:
        columns_data.append({
            "id": column.id,
            "name": column.name,
            "model_id": column.model_id,
            "model_name": column.model.name,
            "project_name": column.model.project.name,
            "data_type": column.data_type,
            "description": column.description,
            "is_primary_key": column.is_primary_key,
            "is_foreign_key": column.is_foreign_key,
            "created_at": column.created_at.isoformat() if column.created_at else None,
            "updated_at": column.updated_at.isoformat() if column.updated_at else None
        })
    
    lineage_data = []
    for edge in lineage:
        lineage_data.append({
            "id": edge.id,
            "upstream_id": edge.upstream_id,
            "upstream_model": edge.upstream_model.name,
            "upstream_project": edge.upstream_model.project.name,
            "downstream_id": edge.downstream_id,
            "downstream_model": edge.downstream_model.name,
            "downstream_project": edge.downstream_model.project.name
        })
    
    return {
        "projects": projects_data,
        "models": models_data,
        "columns": columns_data,
        "lineage": lineage_data
    }

@router.get("/json")
def export_metadata_json(db: Session = Depends(get_db)):
    """Export all metadata as a downloadable JSON file"""
    export_data = export_metadata(db)
    
    # Return as a downloadable JSON file
    return {
        "filename": "dbt_metadata_export.json",
        "content": json.dumps(export_data, indent=2)
    }

@router.get("/yaml")
def export_metadata_yaml(db: Session = Depends(get_db)):
    """Export all metadata as a downloadable YAML file"""
    try:
        import yaml
        
        export_data = export_metadata(db)
        
        # Return as a downloadable YAML file
        return {
            "filename": "dbt_metadata_export.yaml",
            "content": yaml.dump(export_data, sort_keys=False)
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="YAML export requires PyYAML package")

@router.get("/project/{project_id}")
def export_project_metadata(project_id: int, db: Session = Depends(get_db)):
    """Export metadata for a specific project"""
    
    project = db.query(DBProject).filter(DBProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    models = db.query(DBModel).filter(DBModel.project_id == project_id).all()
    model_ids = [model.id for model in models]
    
    columns = db.query(DBColumn).filter(DBColumn.model_id.in_(model_ids)).all()
    
    lineage = db.query(DBLineage).filter(
        (DBLineage.upstream_id.in_(model_ids)) | 
        (DBLineage.downstream_id.in_(model_ids))
    ).all()
    
    # Create serializable data structures
    project_data = {
        "id": project.id,
        "name": project.name,
        "path": project.path,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None
    }
    
    models_data = []
    for model in models:
        models_data.append({
            "id": model.id,
            "name": model.name,
            "file_path": model.file_path,
            "schema": model.schema,
            "materialized": model.materialized,
            "description": model.description,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None
        })
    
    columns_data = []
    for column in columns:
        columns_data.append({
            "id": column.id,
            "name": column.name,
            "model_id": column.model_id,
            "model_name": column.model.name,
            "data_type": column.data_type,
            "description": column.description,
            "is_primary_key": column.is_primary_key,
            "is_foreign_key": column.is_foreign_key,
            "created_at": column.created_at.isoformat() if column.created_at else None,
            "updated_at": column.updated_at.isoformat() if column.updated_at else None
        })
    
    lineage_data = []
    for edge in lineage:
        lineage_data.append({
            "id": edge.id,
            "upstream_id": edge.upstream_id,
            "upstream_model": edge.upstream_model.name,
            "upstream_project": edge.upstream_model.project.name,
            "downstream_id": edge.downstream_id,
            "downstream_model": edge.downstream_model.name,
            "downstream_project": edge.downstream_model.project.name
        })
    
    return {
        "project": project_data,
        "models": models_data,
        "columns": columns_data,
        "lineage": lineage_data
    } 