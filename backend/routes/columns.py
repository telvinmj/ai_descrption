from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import or_, func

from backend.models.schema import Column, ColumnWithRelations
from backend.models.database import (
    ColumnModel as DBColumn,  # Updated import
    Model as DBModel, 
    Project as DBProject
)
from backend.services.database import get_db

router = APIRouter(
    prefix="/columns",
    tags=["columns"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[ColumnWithRelations])
def get_columns(
    model_id: Optional[int] = None,
    project_id: Optional[int] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all columns with optional filtering by model, project, and search term
    """
    query = db.query(
        DBColumn,
        DBModel.name.label("model_name"),
        DBProject.name.label("project_name")
    ).join(DBModel).join(DBProject)
    
    if model_id:
        query = query.filter(DBColumn.model_id == model_id)
    
    if project_id:
        query = query.filter(DBModel.project_id == project_id)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                DBColumn.name.ilike(search_term),
                DBColumn.description.ilike(search_term)
                # DBColumn.ai_description.ilike(search_term)  # Comment out AI field
            )
        )
    
    results = query.all()
    
    columns_with_relations = []
    for result in results:
        column = result[0]
        model_name = result[1]
        project_name = result[2]
        
        column_dict = {
            "id": column.id,
            "name": column.name,
            "model_id": column.model_id,
            "data_type": column.data_type,
            "description": column.description,
            # "ai_description": column.ai_description,  # Comment out AI field
            "is_primary_key": column.is_primary_key,
            "is_foreign_key": column.is_foreign_key,
            "created_at": column.created_at,
            "updated_at": column.updated_at,
            "tags": column.tags,
            "model_name": model_name,
            "project_name": project_name
        }
        
        columns_with_relations.append(ColumnWithRelations(**column_dict))
    
    return columns_with_relations

@router.get("/{column_id}", response_model=ColumnWithRelations)
def get_column(column_id: int, db: Session = Depends(get_db)):
    """Get a specific column by ID with related model and project names"""
    result = db.query(
        DBColumn,
        DBModel.name.label("model_name"),
        DBProject.name.label("project_name")
    ).join(DBModel).join(DBProject).filter(DBColumn.id == column_id).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Column not found")
    
    column, model_name, project_name = result
    
    column_dict = {
        "id": column.id,
        "name": column.name,
        "model_id": column.model_id,
        "data_type": column.data_type,
        "description": column.description,
        # "ai_description": column.ai_description,  # Comment out AI field
        "is_primary_key": column.is_primary_key,
        "is_foreign_key": column.is_foreign_key,
        "created_at": column.created_at,
        "updated_at": column.updated_at,
        "tags": column.tags,
        "model_name": model_name,
        "project_name": project_name
    }
    
    return ColumnWithRelations(**column_dict)

@router.get("/search/related")
def find_related_columns(
    column_name: str = Query(..., description="Column name to search for"),
    db: Session = Depends(get_db)
):
    """
    Find columns with the same name across different models and projects
    """
    columns = db.query(
        DBColumn,
        DBModel.name.label("model_name"),
        DBProject.name.label("project_name")
    ).join(DBModel).join(DBProject).filter(
        func.lower(DBColumn.name) == func.lower(column_name)
    ).all()
    
    results = []
    for column, model_name, project_name in columns:
        results.append({
            "id": column.id,
            "name": column.name,
            "model_id": column.model_id,
            "model_name": model_name,
            "project_name": project_name,
            "description": column.description
            # "ai_description": column.ai_description  # Comment out AI field
        })
    
    return results 