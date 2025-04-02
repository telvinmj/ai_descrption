import axios from 'axios';
import { 
  Project, 
  ProjectSummary, 
  Model, 
  ModelSummary, 
  Column, 
  ColumnWithRelations, 
  UserCorrection,
  ModelWithLineage
} from '../types';

// Clean any quotation marks from the API URL
const rawApiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_URL = rawApiUrl.replace(/"/g, '');

console.log('Using API URL:', API_URL);

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Projects
export const getProjects = async (): Promise<Project[]> => {
  const response = await api.get<Project[]>('/api/projects');
  return response.data;
};

export const getProjectSummaries = async (): Promise<ProjectSummary[]> => {
  const response = await api.get<ProjectSummary[]>('/projects/summary');
  return response.data;
};

export const getProject = async (id: string): Promise<Project> => {
  const response = await api.get<Project>(`/projects/${id}`);
  return response.data;
};

export const createProject = async (project: Partial<Project>): Promise<Project> => {
  const response = await api.post<Project>('/projects', project);
  return response.data;
};

export const refreshProject = async (id: string): Promise<Project> => {
  const response = await api.post<Project>(`/projects/${id}/refresh`);
  return response.data;
};

// Models
export const getModels = async (
  projectId?: string,
  search?: string,
  tag?: string,
  materialized?: string
): Promise<Model[]> => {
  let url = '/api/models';
  const params: Record<string, string> = {};
  
  if (projectId) {
    params.project_id = projectId;
  }
  
  if (search) {
    // Ensure search is trimmed and non-empty
    const trimmedSearch = search.trim();
    if (trimmedSearch) {
      params.search = trimmedSearch;
      console.log('API search param:', trimmedSearch);
    }
  }
  
  if (tag) {
    params.tag = tag;
  }
  
  if (materialized) {
    params.materialized = materialized;
  }
  
  console.log('Calling API with params:', params);
  const response = await api.get(url, { params });
  console.log(`API returned ${response.data.length} models`);
  return response.data;
};

export const getModelSummaries = async (): Promise<ModelSummary[]> => {
  const response = await api.get<ModelSummary[]>('/models/summary');
  return response.data;
};

export const getModel = async (id: string): Promise<Model> => {
  // Add timestamp to prevent browser caching
  const timestamp = new Date().getTime();
  const response = await api.get<Model>(`/api/models/${id}?t=${timestamp}`);
  return response.data;
};

export const getModelWithLineage = async (id: string): Promise<ModelWithLineage> => {
  // Add timestamp to prevent browser caching
  const timestamp = new Date().getTime();
  const response = await api.get<ModelWithLineage>(`/api/models/${id}/lineage?t=${timestamp}`);
  return response.data;
};

export const getLineage = async () => {
  const response = await api.get('/api/lineage');
  return response.data;
};

// Description Updates
export const updateModelDescription = async (
  modelId: string,
  description: string
): Promise<any> => {
  const response = await api.post(`/api/models/${modelId}/description`, {
    description
  });
  return response.data;
};

export const updateColumnDescription = async (
  modelId: string,
  columnName: string,
  description: string
): Promise<any> => {
  const response = await api.post(`/api/columns/${modelId}/${columnName}/description`, {
    description
  });
  return response.data;
};

// Uncertain models and descriptions
export const getUncertainModels = async (): Promise<any> => {
  const response = await api.get('/api/models/uncertain/');
  return response.data;
};

// Columns
export const getColumns = async (
  modelId?: string,
  projectId?: string,
  search?: string
): Promise<ColumnWithRelations[]> => {
  const params = { model_id: modelId, project_id: projectId, search };
  const response = await api.get<ColumnWithRelations[]>('/columns', { params });
  return response.data;
};

export const getColumn = async (id: string): Promise<ColumnWithRelations> => {
  const response = await api.get<ColumnWithRelations>(`/columns/${id}`);
  return response.data;
};

export const getRelatedColumns = async (columnName: string): Promise<ColumnWithRelations[]> => {
  const response = await api.get<ColumnWithRelations[]>('/columns/search/related', {
    params: { column_name: columnName }
  });
  return response.data;
};

// User Corrections
export const createUserCorrection = async (
  entityType: 'model' | 'column',
  entityId: string,
  correctedDescription: string
): Promise<UserCorrection> => {
  if (entityType === 'model') {
    return updateModelDescription(entityId, correctedDescription);
  } else if (entityType === 'column') {
    // For column corrections, entityId is expected to be "modelId:columnName"
    const [modelId, columnName] = entityId.split(':');
    if (!modelId || !columnName) {
      throw new Error(`Invalid column entity ID format: ${entityId}. Expected "modelId:columnName"`);
    }
    return updateColumnDescription(modelId, columnName, correctedDescription);
  }
  
  // Fallback to old API for backward compatibility
  const response = await api.post<UserCorrection>('/corrections', {
    entity_type: entityType,
    entity_id: entityId,
    corrected_description: correctedDescription
  });
  return response.data;
};

// Export
export const exportMetadata = async (format: string = 'json'): Promise<any> => {
  const response = await api.get(`/export/${format}`);
  return response.data;
};

export const exportMetadataToJson = async (): Promise<void> => {
  window.open(`${API_URL}/api/export/json`, '_blank');
};

export const exportMetadataToYaml = async (): Promise<void> => {
  window.open(`${API_URL}/api/export/yaml`, '_blank');
};

// Initialize Database
export const initializeDatabase = async (): Promise<any> => {
  const response = await api.post('/initialize');
  return response.data;
};

export const refreshMetadata = async () => {
  const response = await api.post('/api/refresh');
  return response.data;
};

export const refreshModelMetadata = async (modelId: string, force_update: boolean = true) => {
  try {
    console.log(`API call: Refreshing model ${modelId} with force_update=${force_update}`);
    
    // Add timestamp to prevent caching issues
    const timestamp = new Date().getTime();
    const response = await api.post(`/api/models/${modelId}/refresh?t=${timestamp}`, {
      force_update: force_update
    });
    
    if (!response.data || response.status !== 200) {
      console.error('Invalid response from model refresh API:', response);
      throw new Error('Failed to refresh model: Invalid response');
    }
    
    console.log(`API success: Model ${modelId} refreshed successfully`, response.data);
    
    // After the refresh, fetch the model again to ensure we have the latest data
    console.log(`Fetching latest model data after refresh`);
    const updatedModelResponse = await api.get(`/api/models/${modelId}?t=${timestamp}`);
    
    if (!updatedModelResponse.data) {
      console.warn('Could not fetch updated model data after refresh');
      return response.data; // Return original response if refetch fails
    }
    
    console.log(`Successfully fetched updated model data`);
    return updatedModelResponse.data;
  } catch (error) {
    console.error('Error in refreshModelMetadata:', error);
    throw error;
  }
};

export const refreshModelMetadataWithContext = async (modelId: string, additionalContext: string, force_update: boolean = true) => {
  try {
    console.log(`API call: Refreshing model ${modelId} with context and force_update=${force_update}`);
    
    // Add timestamp to prevent caching issues
    const timestamp = new Date().getTime();
    const response = await api.post(`/api/models/${modelId}/refresh?t=${timestamp}`, {
      additional_prompt: additionalContext,
      force_update: force_update
    });
    
    if (!response.data || response.status !== 200) {
      console.error('Invalid response from model refresh with context API:', response);
      throw new Error('Failed to refresh model with context: Invalid response');
    }
    
    console.log(`API success: Model ${modelId} refreshed with context successfully`, response.data);
    
    // After the refresh, fetch the model again to ensure we have the latest data
    console.log(`Fetching latest model data after context-based refresh`);
    const updatedModelResponse = await api.get(`/api/models/${modelId}?t=${timestamp}`);
    
    if (!updatedModelResponse.data) {
      console.warn('Could not fetch updated model data after context-based refresh');
      return response.data; // Return original response if refetch fails
    }
    
    console.log(`Successfully fetched updated model data after context-based refresh`);
    return updatedModelResponse.data;
  } catch (error) {
    console.error('Error in refreshModelMetadataWithContext:', error);
    throw error;
  }
};

// Watcher service
export const getWatcherStatus = async (): Promise<any> => {
  const response = await api.get('/api/watcher/status');
  return response.data;
};

export const toggleWatcher = async (enable: boolean): Promise<any> => {
  const response = await api.post('/api/watcher/toggle', null, {
    params: { enable }
  });
  return response.data;
};

export default {
  getProjects,
  getModels,
  getModel,
  getModelWithLineage,
  getLineage,
  updateModelDescription,
  updateColumnDescription,
  createUserCorrection,
  refreshMetadata,
  refreshModelMetadata,
  refreshModelMetadataWithContext,
  exportMetadataToJson,
  exportMetadataToYaml,
  getWatcherStatus,
  toggleWatcher,
  getUncertainModels,
}; 