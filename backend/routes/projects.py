from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.models.schema import Project, ProjectCreate, ProjectSummary
from backend.models.database import Project as DBProject
from backend.services.database import get_db
from backend.services.dbt_parser import parse_dbt_manifest

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[ProjectSummary])
def get_projects(db: Session = Depends(get_db)):
    """Get all projects with a count of their models"""
    projects = db.query(DBProject).all()
    
    result = []
    for project in projects:
        model_count = len(project.models)
        result.append(ProjectSummary(
            id=project.id,
            name=project.name,
            model_count=model_count
        ))
    
    return result

@router.get("/{project_id}", response_model=Project)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get a specific project by ID"""
    project = db.query(DBProject).filter(DBProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.post("/", response_model=Project)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    db_project = DBProject(**project.dict())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.post("/{project_id}/refresh", response_model=Project)
def refresh_project(project_id: int, db: Session = Depends(get_db)):
    """Refresh a project by parsing its dbt manifest file again"""
    project = db.query(DBProject).filter(DBProject.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        parse_dbt_manifest(project.path, db)
        db.refresh(project)
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing project: {str(e)}") 