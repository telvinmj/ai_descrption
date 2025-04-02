import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  Typography,
  Card,
  Tabs,
  Table,
  Tag,
  Space,
  Button,
  Spin,
  Alert,
} from 'antd';
import {
  DatabaseOutlined,
  ArrowLeftOutlined,
  RobotOutlined,
  UserOutlined,
  FileOutlined,
} from '@ant-design/icons';
import { getModel, getModelWithLineage } from '../services/api';
import LineageGraph from '../components/LineageGraph';
import ModelDetailView from '../components/ModelDetailView';
import RelatedModels from '../components/RelatedModels';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface LineageLink {
  source: string;
  target: string;
}

interface Model {
  id: string;
  name: string;
  project: string;
  description: string | null;
  ai_description?: string | null;
  user_edited?: boolean;
  columns: any[];
  sql: string | null;
  file_path?: string;
  materialized?: string;
  schema?: string;
  database?: string;
}

interface ModelWithLineage extends Model {
  upstream: Model[];
  downstream: Model[];
}

const ModelDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [model, setModel] = useState<Model | null>(null);
  const [lineageData, setLineageData] = useState<ModelWithLineage | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<string>("details");
  const navigate = useNavigate();

  const fetchModel = async () => {
    try {
      setLoading(true);
      const modelId = id as string;
      const modelData = await getModel(modelId);
      setModel(ensureModelDefaults(modelData) as Model);
      
      // Get lineage data using getModelWithLineage
      try {
        const modelLineage = await getModelWithLineage(modelId);
        // Convert the API model to the component's expected model format
        const convertedLineage = {
          ...ensureModelDefaults(modelLineage),
          upstream: modelLineage.upstream?.map(m => ensureModelDefaults(m)) || [],
          downstream: modelLineage.downstream?.map(m => ensureModelDefaults(m)) || []
        };
        setLineageData(convertedLineage as ModelWithLineage);
      } catch (lineageError) {
        console.error('Error fetching lineage:', lineageError);
      }
      
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error fetching model details');
      console.error('Error fetching model:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModel();
  }, [id, refreshKey]);

  const handleDescriptionUpdated = () => {
    console.log('Model description updated, refreshing model data...');
    
    // First, clear any existing model data to ensure a clean reload
    setModel(null);
    setLineageData(null);
    
    // Set loading state to show loading indicator
    setLoading(true);
    
    // Small delay to ensure UI updates before triggering the reload
    setTimeout(() => {
      console.log('Triggering model data reload...');
      // Increment refresh key to trigger useEffect and reload all model data
      setRefreshKey(prev => prev + 1);
    }, 100);
  };

  const handleTabChange = (key: string) => {
    setActiveTab(key);
  };

  const handleBackClick = () => {
    navigate('/models');
  };

  // Ensure model has all required fields with defaults
  const ensureModelDefaults = (modelData: any) => {
    if (!modelData) return null;
    
    // Define default values for common fields that might be missing
    const defaults = {
      description: "",
      schema: "default",
      materialized: "view",
      file_path: "N/A",
      columns: [],
      sql: "",
      tags: []
    };
    
    // Apply defaults for missing or null fields
    Object.entries(defaults).forEach(([key, value]) => {
      if (!(key in modelData) || modelData[key] === null) {
        modelData[key] = value;
      }
    });
    
    return modelData;
  };

  if (loading) {
    return (
      <div className="loading-container">
        <Spin size="large" tip="Loading model details..." />
      </div>
    );
  }

  if (error) {
    return <Alert type="error" message={error} />;
  }

  if (!model) {
    return <Alert type="warning" message="Model not found" />;
  }

  return (
    <div className="model-detail-container">
      <div className="card-title">
        <Space direction="vertical" size={0} style={{ maxWidth: '100%' }}>
          <Space>
            <Button 
              type="link"
              icon={<ArrowLeftOutlined />} 
              onClick={handleBackClick}
            >
              Back to Models
            </Button>
            <Title level={2} style={{ margin: 0 }}>
              {model.name}
            </Title>
          </Space>
          {model.file_path && model.file_path !== 'N/A' && (
            <div className="file-path-header">
              <FileOutlined style={{ marginRight: '8px' }} /> {model.file_path}
            </div>
          )}
        </Space>
        <Space>
          {model.project && (
            <Tag color="blue" icon={<DatabaseOutlined />} style={{ fontSize: '14px', padding: '4px 8px' }}>
              {model.project}
            </Tag>
          )}
          {model.schema && model.schema !== 'N/A' && (
            <Tag color="green" style={{ fontSize: '14px', padding: '4px 8px' }}>
              {model.database ? `${model.database}.` : ''}{model.schema}
            </Tag>
          )}
          {model.materialized && model.materialized !== 'N/A' && (
            <Tag color="purple" style={{ fontSize: '14px', padding: '4px 8px' }}>
              {model.materialized}
            </Tag>
          )}
        </Space>
      </div>

      <Tabs activeKey={activeTab} onChange={handleTabChange} type="card">
        <TabPane tab="Details" key="details">
          <ModelDetailView model={model} onDescriptionUpdated={handleDescriptionUpdated} />
          
          {/* Add RelatedModels component when lineage data is available */}
          {lineageData && (
            <RelatedModels 
              model={model}
              upstream={lineageData.upstream}
              downstream={lineageData.downstream}
            />
          )}
        </TabPane>
        
        <TabPane tab="Lineage" key="lineage">
          {lineageData ? (
            <div>
              {lineageData.upstream.length === 0 && lineageData.downstream.length === 0 ? (
                <Alert
                  message="No lineage data available"
                  description="This model doesn't have any upstream or downstream dependencies."
                  type="info"
                />
              ) : (
                <div>
                  <h3>Lineage Diagram</h3>
                  <Card className="lineage-graph-container">
                    <LineageGraph 
                      models={[
                        lineageData, 
                        ...lineageData.upstream, 
                        ...lineageData.downstream
                      ]}
                      lineage={[
                        ...lineageData.upstream.map(m => ({ source: m.id, target: lineageData.id })),
                        ...lineageData.downstream.map(m => ({ source: lineageData.id, target: m.id }))
                      ]} 
                    />
                  </Card>

                  {lineageData.upstream.length > 0 && (
                    <div style={{ marginTop: '24px' }}>
                      <h3>Upstream Models (Sources)</h3>
                      <Table
                        dataSource={lineageData.upstream}
                        columns={[
                          { title: 'Name', dataIndex: 'name', key: 'name', 
                            render: (text, record: any) => (
                              <Link to={`/models/${record.id}`} onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/models/${record.id}`);
                              }}>{text}</Link>
                            )
                          },
                          { title: 'Project', dataIndex: 'project', key: 'project' },
                          { title: 'Description', dataIndex: 'description', key: 'description',
                            render: (text, record: any) => (
                              <div>
                                {text || 'No description'}
                                {record.ai_description && !record.user_edited && (
                                  <Tag color="blue" icon={<RobotOutlined />} style={{ marginLeft: 8 }}>
                                    AI Generated
                                  </Tag>
                                )}
                                {record.user_edited && (
                                  <Tag color="green" icon={<UserOutlined />} style={{ marginLeft: 8 }}>
                                    User Edited
                                  </Tag>
                                )}
                              </div>
                            )
                          },
                        ]}
                        pagination={false}
                      />
                    </div>
                  )}

                  {lineageData.downstream.length > 0 && (
                    <div style={{ marginTop: '24px' }}>
                      <h3>Downstream Models (Consumers)</h3>
                      <Table
                        dataSource={lineageData.downstream}
                        columns={[
                          { title: 'Name', dataIndex: 'name', key: 'name', 
                            render: (text, record: any) => (
                              <Link to={`/models/${record.id}`} onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/models/${record.id}`);
                              }}>{text}</Link>
                            )
                          },
                          { title: 'Project', dataIndex: 'project', key: 'project' },
                          { title: 'Description', dataIndex: 'description', key: 'description',
                            render: (text, record: any) => (
                              <div>
                                {text || 'No description'}
                                {record.ai_description && !record.user_edited && (
                                  <Tag color="blue" icon={<RobotOutlined />} style={{ marginLeft: 8 }}>
                                    AI Generated
                                  </Tag>
                                )}
                              </div>
                            )
                          },
                        ]}
                        pagination={false}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <Alert
              message="Loading lineage data..."
              description="Please wait while we load the lineage information."
              type="info"
            />
          )}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ModelDetail; 