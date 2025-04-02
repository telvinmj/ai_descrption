from sqlalchemy import Column as SQLColumn, String, Integer, ForeignKey, Text, DateTime, Table, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

# Association tables
model_tag = Table(
    'model_tag', Base.metadata,
    SQLColumn('model_id', Integer, ForeignKey('models.id')),
    SQLColumn('tag_id', Integer, ForeignKey('tags.id'))
)

column_tag = Table(
    'column_tag', Base.metadata,
    SQLColumn('column_id', Integer, ForeignKey('columns.id')),
    SQLColumn('tag_id', Integer, ForeignKey('tags.id'))
)

class Project(Base):
    __tablename__ = 'projects'
    
    id = SQLColumn(Integer, primary_key=True)
    name = SQLColumn(String(100), nullable=False)
    path = SQLColumn(String(255), nullable=False)
    created_at = SQLColumn(DateTime, default=datetime.datetime.utcnow)
    updated_at = SQLColumn(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    models = relationship("Model", back_populates="project")
    
    def __repr__(self):
        return f"<Project(name='{self.name}')>"

class Model(Base):
    __tablename__ = 'models'
    
    id = SQLColumn(Integer, primary_key=True)
    name = SQLColumn(String(100), nullable=False)
    project_id = SQLColumn(Integer, ForeignKey('projects.id'))
    file_path = SQLColumn(String(255), nullable=False)
    schema = SQLColumn(String(100))
    materialized = SQLColumn(String(50))
    description = SQLColumn(Text)
    ai_description = SQLColumn(Text)  # AI-generated description
    user_edited = SQLColumn(Boolean, default=False)  # Track if description was edited by user
    raw_sql = SQLColumn(Text)
    compiled_sql = SQLColumn(Text)
    created_at = SQLColumn(DateTime, default=datetime.datetime.utcnow)
    updated_at = SQLColumn(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    project = relationship("Project", back_populates="models")
    columns = relationship("ColumnModel", back_populates="model")
    tags = relationship("Tag", secondary=model_tag, back_populates="models")
    upstream_edges = relationship("Lineage", foreign_keys="[Lineage.downstream_id]", back_populates="downstream_model")
    downstream_edges = relationship("Lineage", foreign_keys="[Lineage.upstream_id]", back_populates="upstream_model")
    
    def __repr__(self):
        return f"<Model(name='{self.name}')>"

class ColumnModel(Base):
    __tablename__ = 'columns'
    
    id = SQLColumn(Integer, primary_key=True)
    name = SQLColumn(String(100), nullable=False)
    model_id = SQLColumn(Integer, ForeignKey('models.id'))
    data_type = SQLColumn(String(50))
    description = SQLColumn(Text)
    ai_description = SQLColumn(Text)  # AI-generated description
    user_edited = SQLColumn(Boolean, default=False)  # Track if description was edited by user
    is_primary_key = SQLColumn(Boolean, default=False)
    is_foreign_key = SQLColumn(Boolean, default=False)
    created_at = SQLColumn(DateTime, default=datetime.datetime.utcnow)
    updated_at = SQLColumn(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    model = relationship("Model", back_populates="columns")
    tags = relationship("Tag", secondary=column_tag, back_populates="columns")
    
    def __repr__(self):
        return f"<Column(name='{self.name}')>"

class Lineage(Base):
    __tablename__ = 'lineage'
    
    id = SQLColumn(Integer, primary_key=True)
    upstream_id = SQLColumn(Integer, ForeignKey('models.id'))
    downstream_id = SQLColumn(Integer, ForeignKey('models.id'))
    
    upstream_model = relationship("Model", foreign_keys=[upstream_id], back_populates="downstream_edges")
    downstream_model = relationship("Model", foreign_keys=[downstream_id], back_populates="upstream_edges")
    
    def __repr__(self):
        return f"<Lineage(upstream='{self.upstream_id}', downstream='{self.downstream_id}')>"

class Tag(Base):
    __tablename__ = 'tags'
    
    id = SQLColumn(Integer, primary_key=True)
    name = SQLColumn(String(50), nullable=False, unique=True)
    
    models = relationship("Model", secondary=model_tag, back_populates="tags")
    columns = relationship("ColumnModel", secondary=column_tag, back_populates="tags")
    
    def __repr__(self):
        return f"<Tag(name='{self.name}')>"

class UserCorrection(Base):
    __tablename__ = 'user_corrections'
    
    id = SQLColumn(Integer, primary_key=True)
    entity_type = SQLColumn(String(50), nullable=False)  # 'model' or 'column'
    entity_id = SQLColumn(Integer, nullable=False)
    original_description = SQLColumn(Text)
    corrected_description = SQLColumn(Text, nullable=False)
    created_at = SQLColumn(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<UserCorrection(entity_id='{self.entity_id}', entity_type='{self.entity_type}')>" 