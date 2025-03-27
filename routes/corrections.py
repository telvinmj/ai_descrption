from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.models.schema import UserCorrection, UserCorrectionCreate
from backend.models.database import (
    UserCorrection as DBUserCorrection,
    Model as DBModel, 
    ColumnModel as DBColumn  # Updated import
)
from backend.services.database import get_db

router = APIRouter(
    prefix="/corrections",
    tags=["corrections"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[UserCorrection])
def get_corrections(db: Session = Depends(get_db)):
    """Get all user corrections"""
    return db.query(DBUserCorrection).all()

@router.post("/", response_model=UserCorrection)
def create_correction(correction: UserCorrectionCreate, db: Session = Depends(get_db)):
    """Create a new user correction and update the corresponding entity"""
    # Validate entity exists
    if correction.entity_type == 'model':
        entity = db.query(DBModel).filter(DBModel.id == correction.entity_id).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Store original description
        original = entity.ai_description if entity.ai_description else entity.description
        
        # Update model with corrected description
        entity.description = correction.corrected_description
        entity.user_edited = True  # Mark as user-edited
        
    elif correction.entity_type == 'column':
        entity = db.query(DBColumn).filter(DBColumn.id == correction.entity_id).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Column not found")
        
        # Store original description
        original = entity.ai_description if entity.ai_description else entity.description
        
        # Update column with corrected description
        entity.description = correction.corrected_description
        entity.user_edited = True  # Mark as user-edited
        
    else:
        raise HTTPException(status_code=400, detail="Invalid entity type")
    
    # Create correction record
    db_correction = DBUserCorrection(
        entity_type=correction.entity_type,
        entity_id=correction.entity_id,
        original_description=original,
        corrected_description=correction.corrected_description
    )
    
    db.add(db_correction)
    db.commit()
    db.refresh(db_correction)
    
    return db_correction 