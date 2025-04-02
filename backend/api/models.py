from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..services.metadata_service import MetadataService

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/")
async def get_models(
    search: Optional[str] = None,
    project_id: Optional[str] = None,
    include_lineage: bool = False,
    limit: int = Query(100, gt=0, le=1000)
):
    """
    Get all models with optional filtering
    
    Args:
        search: Optional search term to filter models
        project_id: Optional project ID to filter models
        include_lineage: Whether to include lineage information
        limit: Maximum number of models to return
    
    Returns:
        List of models
    """
    metadata_service = MetadataService()
    
    if include_lineage:
        # Get models with lineage
        models = []
        base_models = metadata_service.get_models(project_id, search)
        
        # Apply limit
        base_models = base_models[:limit]
        
        # Add lineage to each model
        for model in base_models:
            model_with_lineage = metadata_service.get_model_with_lineage(model["id"])
            if model_with_lineage:
                models.append(model_with_lineage)
    else:
        # Get models without lineage
        models = metadata_service.get_models(project_id, search)
        
        # Apply limit
        models = models[:limit]
    
    return models

@router.get("/{model_id}")
async def get_model(model_id: str):
    """
    Get a specific model by ID
    
    Args:
        model_id: ID of the model to retrieve
    
    Returns:
        Model details
    """
    metadata_service = MetadataService()
    model = metadata_service.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model

@router.get("/{model_id}/lineage")
async def get_model_lineage(model_id: str):
    """
    Get a model with its lineage information
    
    Args:
        model_id: ID of the model to retrieve lineage for
    
    Returns:
        Model with upstream and downstream lineage
    """
    if model_id == "NaN" or not model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    
    metadata_service = MetadataService()
    model = metadata_service.get_model_with_lineage(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model

# Comment out or remove the AI suggestions endpoint
# @router.get("/{model_id}/suggestions", response_model=List[ModelSuggestion])
# async def get_model_suggestions(model_id: str, db: Session = Depends(get_db_session)):
#     """Get AI-generated suggestions for a model"""
#     model_service = ModelService(db)
#     suggestions = model_service.get_model_suggestions(model_id)
#     
#     return suggestions 