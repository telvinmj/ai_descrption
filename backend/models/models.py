from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class Model(BaseModel):
    id: int
    name: str
    project_id: int
    project_name: Optional[str]
    description: Optional[str]
    # ai_description: Optional[str]  # Commented out AI field
    raw_sql: Optional[str]
    compiled_sql: Optional[str]
    column_count: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class RelatedModel(BaseModel):
    id: str
    name: str
    project_name: str
    is_source: Optional[bool] = False

class ModelWithLineage(Model):
    upstream: List[RelatedModel] = []
    downstream: List[RelatedModel] = []

# Comment out AI-related model
# class ModelSuggestion(BaseModel):
#     id: int
#     model_id: int
#     suggestion: str
#     created_at: datetime 