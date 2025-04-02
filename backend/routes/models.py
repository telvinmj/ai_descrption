from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import or_, func

from backend.models.schema import Model, ModelSummary
from backend.models.database import Model as DBModel, Project as DBProject
from backend.services.database import get_db
# from backend.services.ai_service import get_model_suggestions  # Comment out AI import
# from backend.models.models import ModelSuggestion  # Commented out AI-related import
from backend.services.model_service import ModelService

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[ModelSummary])
def get_models(
    project_id: Optional[int] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all models with optional filtering by project and search term
    """
    query = db.query(
        DBModel.id, 
        DBModel.name, 
        DBModel.project_id, 
        DBModel.schema_name,
        DBProject.name.label("project_name"),
        func.count(DBModel.columns).label("column_count")
    ).join(DBProject)
    
    if project_id:
        query = query.filter(DBModel.project_id == project_id)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                DBModel.name.ilike(search_term),
                DBModel.description.ilike(search_term)
                # DBModel.ai_description.ilike(search_term)  # Comment out AI field
            )
        )
    
    # Group by to make the count work
    query = query.group_by(
        DBModel.id, 
        DBModel.name, 
        DBModel.project_id, 
        DBModel.schema_name,
        DBProject.name
    )
    
    models = query.all()
    
    result = []
    for model in models:
        result.append(ModelSummary(
            id=model.id,
            name=model.name,
            project_id=model.project_id,
            project_name=model.project_name,
            schema=model.schema_name,
            column_count=model.column_count
        ))
    
    return result

@router.get("/{model_id}", response_model=Model)
def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific model by ID"""
    model = db.query(DBModel).filter(DBModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

# Comment out AI-related route
# @router.get("/{model_id}/suggestions", response_model=List[ModelSuggestion])
# def get_model_suggestions(
#     model_id: int,
#     db: Session = Depends(get_db)
# ):
#     """Get AI-generated suggestions for a model"""
#     service = ModelService(db)
#     return service.get_model_suggestions(model_id)

@router.get("/search/lineage")
def search_models_with_lineage(
    search: str = Query(..., description="Search term for models"),
    db: Session = Depends(get_db)
):
    """
    Search for models and return their lineage information
    """
    search_term = f"%{search}%"
    
    # Find matching models
    models = db.query(DBModel).filter(
        or_(
            DBModel.name.ilike(search_term),
            DBModel.description.ilike(search_term)
            # DBModel.ai_description.ilike(search_term)  # Comment out AI field
        )
    ).all()
    
    result = []
    for model in models:
        # Get upstream models
        upstream = []
        for edge in model.upstream_edges:
            upstream_model = edge.upstream_model
            upstream.append({
                "id": upstream_model.id,
                "name": upstream_model.name,
                "project_name": upstream_model.project.name
            })
        
        # Get downstream models
        downstream = []
        for edge in model.downstream_edges:
            downstream_model = edge.downstream_model
            downstream.append({
                "id": downstream_model.id,
                "name": downstream_model.name,
                "project_name": downstream_model.project.name
            })
        
        result.append({
            "id": model.id,
            "name": model.name,
            "project_name": model.project.name,
            "upstream": upstream,
            "downstream": downstream
        })
    
    return result 