from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from ..models.models import Model, ModelWithLineage, RelatedModel
# from ..models.models import ModelSuggestion  # Commented out AI-related import
from ..db.models import (
    Model as DBModel,
    Project as DBProject,
    ColumnModel,
    Lineage as DBLineage,
    # ModelSuggestion as DBModelSuggestion  # Commented out AI-related import
)

class ModelService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_models(
        self, 
        search: Optional[str] = None, 
        project_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Model]:
        """Get models with optional filtering"""
        query = self.db.query(DBModel)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    DBModel.name.ilike(search_term),
                    DBModel.description.ilike(search_term)
                )
            )
        
        if project_id:
            query = query.filter(DBModel.project_id == project_id)
        
        # Join with project to get project_name
        query = query.join(DBProject, DBModel.project_id == DBProject.id)
        
        # Get column count
        query = query.outerjoin(
            DBColumn, 
            DBModel.id == DBColumn.model_id
        ).add_columns(
            func.count(DBColumn.id).label('column_count')
        ).group_by(DBModel.id, DBProject.id)
        
        # Add project name to results
        query = query.add_columns(DBProject.name.label('project_name'))
        
        # Execute query with limit
        results = query.limit(limit).all()
        
        # Format results
        models = []
        for result in results:
            model_data = result[0].__dict__
            model_data['column_count'] = result[1]
            model_data['project_name'] = result[2]
            
            # Remove SQLAlchemy instance state
            if '_sa_instance_state' in model_data:
                del model_data['_sa_instance_state']
            
            models.append(Model(**model_data))
        
        return models
    
    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Get a single model by ID"""
        result = self.db.query(
            DBModel,
            DBProject.name.label('project_name'),
            func.count(DBColumn.id).label('column_count')
        ).join(
            DBProject, 
            DBModel.project_id == DBProject.id
        ).outerjoin(
            DBColumn,
            DBModel.id == DBColumn.model_id
        ).filter(
            DBModel.id == model_id
        ).group_by(
            DBModel.id, 
            DBProject.id
        ).first()
        
        if not result:
            return None
        
        model_data = result[0].__dict__
        model_data['project_name'] = result[1]
        model_data['column_count'] = result[2]
        
        # Remove SQLAlchemy instance state
        if '_sa_instance_state' in model_data:
            del model_data['_sa_instance_state']
        
        return Model(**model_data)
    
    def get_models_with_lineage(
        self, 
        search: Optional[str] = None, 
        project_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ModelWithLineage]:
        """Get models with lineage information"""
        # First get filtered models
        models = self.get_models(search, project_id, limit)
        
        # Get lineage for each model
        models_with_lineage = []
        for model in models:
            lineage_model = self.get_model_lineage_by_id(model.id)
            if lineage_model:
                models_with_lineage.append(lineage_model)
        
        return models_with_lineage
    
    def get_model_lineage_by_id(self, model_id: str) -> Optional[ModelWithLineage]:
        """Get model with its upstream and downstream lineage"""
        # Get the model first
        model = self.get_model_by_id(model_id)
        if not model:
            return None
        
        # Get upstream models (parents)
        upstream_query = self.db.query(
            DBModel, 
            DBProject.name.label('project_name'),
            func.count(DBColumn.id).label('column_count'),
            DBModelRelation.source_is_source.label('is_source')
        ).join(
            DBProject,
            DBModel.project_id == DBProject.id
        ).outerjoin(
            DBColumn,
            DBModel.id == DBColumn.model_id
        ).join(
            DBModelRelation,
            DBModel.id == DBModelRelation.source_model_id
        ).filter(
            DBModelRelation.target_model_id == model_id
        ).group_by(
            DBModel.id,
            DBProject.id,
            DBModelRelation.source_is_source
        ).all()
        
        # Get downstream models (children)
        downstream_query = self.db.query(
            DBModel,
            DBProject.name.label('project_name'),
            func.count(DBColumn.id).label('column_count')
        ).join(
            DBProject,
            DBModel.project_id == DBProject.id
        ).outerjoin(
            DBColumn,
            DBModel.id == DBColumn.model_id
        ).join(
            DBModelRelation,
            DBModel.id == DBModelRelation.target_model_id
        ).filter(
            DBModelRelation.source_model_id == model_id
        ).group_by(
            DBModel.id,
            DBProject.id
        ).all()
        
        # Format upstream models
        upstream_models = []
        for result in upstream_query:
            model_data = {
                'id': result[0].id,
                'name': result[0].name,
                'project_name': result[1],
                'is_source': result[3]
            }
            upstream_models.append(RelatedModel(**model_data))
        
        # Format downstream models
        downstream_models = []
        for result in downstream_query:
            model_data = {
                'id': result[0].id,
                'name': result[0].name,
                'project_name': result[1]
            }
            downstream_models.append(RelatedModel(**model_data))
        
        # Create the model with lineage
        model_dict = model.dict()
        model_dict['upstream'] = upstream_models
        model_dict['downstream'] = downstream_models
        
        return ModelWithLineage(**model_dict)
    
    # Comment out AI-related method
    # def get_model_suggestions(self, model_id: str) -> List[ModelSuggestion]:
    #     """Get AI-generated suggestions for a model"""
    #     suggestions_query = self.db.query(
    #         DBModelSuggestion
    #     ).filter(
    #         DBModelSuggestion.model_id == model_id
    #     ).all()
    #     
    #     suggestions = []
    #     for suggestion in suggestions_query:
    #         suggestion_data = suggestion.__dict__
    #         
    #         if '_sa_instance_state' in suggestion_data:
    #             del suggestion_data['_sa_instance_state']
    #             
    #         suggestions.append(ModelSuggestion(**suggestion_data))
    #         
    #     return suggestions 