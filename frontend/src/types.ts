import { ReactNode } from "react";

// Basic model types
export interface Column {
  name: string;
  type: string;
  description: string | null;
  ai_description?: string | null;
  user_edited?: boolean;
  isPrimaryKey: boolean;
  isForeignKey: boolean;
  id?: string;
}

export interface RelatedModel {
  id: string;
  name: string;
  project_name: string;
  is_source: boolean;
}

export interface Model {
  id: string;
  name: string;
  project: string;
  description: string | null;
  ai_description?: string | null;
  user_edited?: boolean;
  columns: Column[];
  sql: string | null;
  schema?: string;
  file_path?: string;
  materialized?: string;
  database?: string;
  tags?: string[];
  raw_sql?: string;
  originalModels?: Model[];
  stats?: any;  // Optional statistics from dbt catalog
  catalog_metadata?: any;  // Optional additional catalog metadata
}

export interface ModelWithLineage extends Model {
  upstream: Model[];
  downstream: Model[];
}

// Project types
export interface Project {
  updated_at: string | number | Date;
  created_at: string | number | Date;
  path: ReactNode;
  id: string;
  name: string;
  description: string;
}

export interface ProjectSummary extends Project {
  model_count: number;
}

// Additional types
export interface ModelSummary {
  id: string;
  name: string;
  project_name: string;
  project_id: string;
  description: string | null;
  ai_description?: string | null;
  user_edited?: boolean;
  column_count: number;
}

export interface ColumnWithRelations extends Column {
  id: string;
  model_id: string;
  model_name: string;
  project_id: string;
  project_name: string;
}

export interface UserCorrection {
  id: string;
  target_type: 'model' | 'column';
  target_id: string;
  field: string;
  old_value: string | null;
  new_value: string;
  created_at: string;
}

export interface LineageLink {
  source: string;
  target: string;
} 