export interface Project {
  id: string;
  name: string;
  path: string;
  created_at: string;
  updated_at: string;
  models?: Model[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  model_count: number;
}

export interface Model {
  id: string;
  name: string;
  project_id: string;
  file_path: string;
  schema: string | null;
  materialized: string | null;
  description: string | null;
  ai_description: string | null;
  raw_sql: string | null;
  compiled_sql: string | null;
  created_at: string;
  updated_at: string;
  columns: Column[];
  tags: Tag[];
}

export interface ModelSummary {
  id: string;
  name: string;
  project_id: string;
  project_name: string;
  schema: string | null;
  column_count: number;
}

export interface Column {
  id: string;
  name: string;
  model_id: string;
  data_type: string | null;
  description: string | null;
  ai_description: string | null;
  is_primary_key: boolean;
  is_foreign_key: boolean;
  created_at: string;
  updated_at: string;
  tags: Tag[];
}

export interface ColumnWithRelations extends Column {
  model_name: string;
  project_name: string;
}

export interface Tag {
  id: string;
  name: string;
}

export interface UserCorrection {
  id: string;
  entity_type: 'model' | 'column';
  entity_id: string;
  original_description: string | null;
  corrected_description: string;
  created_at: string;
}

export interface LineageNode {
  id: string;
  name: string;
  project_name: string;
}

export interface ModelWithLineage {
  id: string;
  name: string;
  project_name: string;
  upstream: LineageNode[];
  downstream: LineageNode[];
}

export interface ApiError {
  detail: string;
} 