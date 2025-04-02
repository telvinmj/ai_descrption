import React, { useState } from 'react';
import { Card, Input, Button, List, Spin, Alert, Empty } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import LineageGraph from '../components/LineageGraph';

const { Search } = Input;

// Define types to match LineageGraph component
interface Column {
  name: string;
  type: string;
  description: string | null;
  isPrimaryKey: boolean;
  isForeignKey: boolean;
}

interface Model {
  id: string;
  name: string;
  project: string;
  description: string | null;
  columns: Column[];
  sql: string | null; // This was missing
}

interface LineageLink {
  source: string;
  target: string;
}

// Extend Model to include upstream and downstream
interface ModelWithLineage extends Model {
  upstream: Model[];
  downstream: Model[];
}

const LineageExplorer: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<ModelWithLineage[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelWithLineage | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Create lineage data for visualization from selected model
  const getLineageData = (model: ModelWithLineage): LineageLink[] => {
    const links: LineageLink[] = [];
    
    // Add upstream links
    model.upstream?.forEach(upModel => {
      links.push({
        source: upModel.id,
        target: model.id
      });
    });
    
    // Add downstream links
    model.downstream?.forEach(downModel => {
      links.push({
        source: model.id,
        target: downModel.id
      });
    });
    
    return links;
  };
  
  // Get all models for visualization
  const getModelsForVisualization = (model: ModelWithLineage): Model[] => {
    const models: Model[] = [model];
    
    // Add upstream models
    if (model.upstream) {
      models.push(...model.upstream);
    }
    
    // Add downstream models
    if (model.downstream) {
      models.push(...model.downstream);
    }
    
    return models;
  };

  return (
    <div className="lineage-explorer">
      <h2>Lineage Explorer</h2>
      <p>Search for a model to explore its relationships with other models.</p>
      
      <Search
        placeholder="Search for a model..."
        enterButton={<Button icon={<SearchOutlined />}>Search</Button>}
        size="large"
        loading={loading}
        onSearch={value => {
          setSearchQuery(value);
          // Add search logic here
        }}
        style={{ marginBottom: 20 }}
      />
      
      {loading && <Spin size="large" />}
      
      {error && <Alert type="error" message={error} />}
      
      {searchResults.length > 0 ? (
        <List
          itemLayout="horizontal"
          dataSource={searchResults}
          renderItem={item => (
            <List.Item
              actions={[
                <Button 
                  key="view"
                  onClick={() => setSelectedModel(item)}
                >
                  View Lineage
                </Button>
              ]}
            >
              <List.Item.Meta
                title={item.name}
                description={`${item.project} | ${item.description || 'No description'}`}
              />
            </List.Item>
          )}
        />
      ) : (
        searchQuery && !loading && <Empty description="No models found matching your search." />
      )}
      
      {selectedModel && (
        <Card 
          title={`Lineage for ${selectedModel.name}`} 
          style={{ marginTop: 20 }}
        >
          <LineageGraph 
            models={getModelsForVisualization(selectedModel)} 
            lineage={getLineageData(selectedModel)}
          />
        </Card>
      )}
    </div>
  );
};

export default LineageExplorer; 