import axios from 'axios';

export interface UncertainDescription {
  id: string;
  modelName: string;
  columnName: string;
  currentDescription: string;
  confidenceScore: number;
  uncertaintyReason: string;
}

export interface FeedbackSubmission {
  id: string;
  improvedDescription: string;
  feedback: string;
}

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const uncertainDescriptionService = {
  async getUncertainDescriptions(): Promise<UncertainDescription[]> {
    const response = await axios.get(`${BASE_URL}/api/columns/uncertain`);
    return response.data.columns.map((col: any) => ({
      id: `${col.model_id}:${col.name}`,
      modelName: col.model_name,
      columnName: col.name,
      currentDescription: col.description || '',
      confidenceScore: col.confidence_score || 0,
      uncertaintyReason: col.uncertainty_reason || 'Needs review'
    }));
  },

  async getSampleData(modelName: string, columnName: string): Promise<any[]> {
    // Since we don't have actual sample data, we'll return some mock data
    return [
      { value: 'Example Value 1', count: 100 },
      { value: 'Example Value 2', count: 75 },
      { value: 'Example Value 3', count: 50 }
    ];
  },

  async submitFeedback(feedback: FeedbackSubmission): Promise<void> {
    const [modelId, columnName] = feedback.id.split(':');
    await axios.post(
      `${BASE_URL}/api/columns/${modelId}/${columnName}/improve`,
      {
        feedback: feedback.feedback,
        improvedDescription: feedback.improvedDescription
      }
    );
  }
}; 