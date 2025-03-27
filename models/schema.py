from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class TagBase(BaseModel):
    name: str

class TagCreate(TagBase):
    pass

class Tag(TagBase):
    id: int
    
    class Config:
        orm_mode = True

class ColumnBase(BaseModel):
    name: str
    data_type: Optional[str] = None
    description: Optional[str] = None
    ai_description: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False

class ColumnCreate(ColumnBase):
    model_id: int

class Column(ColumnBase):
    id: int
    model_id: int
    created_at: datetime
    updated_at: datetime
    tags: List[Tag] = []
    
    class Config:
        orm_mode = True

class ModelBase(BaseModel):
    name: str
    file_path: str
    schema: Optional[str] = None
    materialized: Optional[str] = None
    description: Optional[str] = None
    ai_description: Optional[str] = None
    raw_sql: Optional[str] = None
    compiled_sql: Optional[str] = None

class ModelCreate(ModelBase):
    project_id: int

class Model(ModelBase):
    id: int
    project_id: int
    created_at: datetime
    updated_at: datetime
    columns: List[Column] = []
    tags: List[Tag] = []
    
    class Config:
        orm_mode = True

class LineageBase(BaseModel):
    upstream_id: int
    downstream_id: int

class LineageCreate(LineageBase):
    pass

class Lineage(LineageBase):
    id: int
    
    class Config:
        orm_mode = True

class ProjectBase(BaseModel):
    name: str
    path: str

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    created_at: datetime
    updated_at: datetime
    models: List[Model] = []
    
    class Config:
        orm_mode = True

class UserCorrectionBase(BaseModel):
    entity_type: str  # 'model' or 'column'
    entity_id: int
    original_description: Optional[str] = None
    corrected_description: str

class UserCorrectionCreate(UserCorrectionBase):
    pass

class UserCorrection(UserCorrectionBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class ProjectSummary(BaseModel):
    id: int
    name: str
    model_count: int
    
    class Config:
        orm_mode = True

class ModelSummary(BaseModel):
    id: int
    name: str
    project_id: int
    project_name: str
    schema: Optional[str] = None
    column_count: int
    
    class Config:
        orm_mode = True

class ColumnWithRelations(Column):
    model_name: str
    project_name: str
    tags: Optional[List[Any]]
    
    class Config:
        orm_mode = True

class MetadataExport(BaseModel):
    projects: List[Dict[str, Any]]
    models: List[Dict[str, Any]]
    columns: List[Dict[str, Any]]
    lineage: List[Dict[str, Any]]

# Comment out AI-related models
# class ModelSuggestion(BaseModel):
#     id: int
#     model_id: int
#     suggestion: str
#     created_at: datetime
#     
#     class Config:
#         orm_mode = True 