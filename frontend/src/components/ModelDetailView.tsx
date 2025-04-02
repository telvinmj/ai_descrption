import React, { useState } from 'react';
import { Card, Descriptions, Table, Tag, Space, Tabs, Button, message, Tooltip, Typography, Input, Modal } from 'antd';
import { Link } from 'react-router-dom';
import DescriptionEdit from './DescriptionEdit';
import { DatabaseOutlined, TableOutlined, SyncOutlined, FileOutlined, FolderOutlined, RobotOutlined, ExclamationCircleOutlined, InfoCircleOutlined, InfoCircleFilled } from '@ant-design/icons';
import { refreshModelMetadata, refreshModelMetadataWithContext } from '../services/api';

const { TabPane } = Tabs;
const { Text } = Typography;
const { TextArea } = Input;

interface Column {
  id?: any;
  name: string;
  type: string;
  description: string | null;
  ai_description?: string | null;
  user_edited?: boolean;
  isPrimaryKey?: boolean;
  isForeignKey?: boolean;
  ai_confidence_score?: number;
}

interface Model {
  id: string;
  name: string;
  project: string;
  description: string | null;
  ai_description?: string | null;
  user_edited?: boolean;
  columns: Column[];
  sql: string | null;
  file_path?: string;
  materialized?: string;
  schema?: string;
  database?: string;
  ai_confidence_score?: number;
}

interface ModelDetailViewProps {
  model: Model;
  onDescriptionUpdated: () => void;
}

const ModelDetailView: React.FC<ModelDetailViewProps> = ({ model, onDescriptionUpdated }) => {
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [isContextModalVisible, setIsContextModalVisible] = useState<boolean>(false);
  const [additionalContext, setAdditionalContext] = useState<string>('');
  
  const handleRefreshMetadata = async () => {
    try {
      setRefreshing(true);
      message.loading({ content: 'Refreshing AI descriptions...', key: 'refresh', duration: 0 });
      console.log(`Starting refresh for model ${model.id} (${model.name}) with force_update=true`);
      
      // Explicitly set force_update to true to ensure descriptions are overwritten
      const result = await refreshModelMetadata(model.id, true);
      console.log('Refresh API response received');
      
      // Log the results to help debug what's happening
      if (result) {
        console.log('Updated model data received:', {
          name: result.name,
          description: result.description?.substring(0, 50) + '...',
          ai_description: result.ai_description?.substring(0, 50) + '...',
          columns_count: result.columns?.length,
          columns_with_desc: result.columns?.filter((c: Column) => c.description).length,
          columns_with_ai_desc: result.columns?.filter((c: Column) => c.ai_description).length
        });
      }
      
      message.success({ content: 'AI descriptions refreshed successfully!', key: 'refresh' });
      
      // Ensure parent component fully refreshes the model data
      setTimeout(() => {
        console.log('Triggering parent component refresh...');
        
        // This function is passed from the parent (ModelDetail component)
        // and triggers a state update to reload the model and lineage data
        onDescriptionUpdated();
        
        // Indicate refresh is complete
        console.log('Refresh complete - UI update requested');
      }, 1000); // Increased delay to ensure backend processing completes
    } catch (error) {
      console.error('Error refreshing model metadata:', error);
      message.error({ 
        content: 'Failed to refresh AI descriptions. Please try again or check console for details.',
        key: 'refresh' 
      });
    } finally {
      setRefreshing(false);
    }
  };

  const showContextModal = () => {
    setIsContextModalVisible(true);
  };

  const handleContextModalCancel = () => {
    setIsContextModalVisible(false);
  };

  const handleRefreshWithContext = async () => {
    try {
      setRefreshing(true);
      setIsContextModalVisible(false);
      
      message.loading({ content: 'Refreshing AI descriptions with additional context...', key: 'refresh', duration: 0 });
      console.log(`Starting context-based refresh for model ${model.id} (${model.name}) with context: ${additionalContext} and force_update=true`);
      
      // Explicitly set force_update to true to ensure descriptions are overwritten
      const result = await refreshModelMetadataWithContext(model.id, additionalContext, true);
      console.log('Context-based refresh API response received');
      
      // Log the results to help debug what's happening
      if (result) {
        console.log('Updated model data after context-based refresh:', {
          name: result.name,
          description: result.description?.substring(0, 50) + '...',
          ai_description: result.ai_description?.substring(0, 50) + '...',
          columns_count: result.columns?.length,
          columns_with_desc: result.columns?.filter((c: Column) => c.description).length,
          columns_with_ai_desc: result.columns?.filter((c: Column) => c.ai_description).length
        });
      }
      
      message.success({ content: 'AI descriptions refreshed successfully with your context!', key: 'refresh' });
      setAdditionalContext(''); // Clear the context after successful refresh
      
      // Ensure parent component fully refreshes the model data
      setTimeout(() => {
        console.log('Triggering parent component refresh after context-based update...');
        onDescriptionUpdated();
        console.log('Context-based refresh complete - UI update requested');
      }, 1000); // Increased delay to ensure backend processing completes
    } catch (error) {
      console.error('Error refreshing model metadata with context:', error);
      message.error({ 
        content: 'Failed to refresh AI descriptions with context. Please try again.',
        key: 'refresh' 
      });
    } finally {
      setRefreshing(false);
    }
  };

  // Format file path for display with highlighting the filename
  const formatFilePath = (path: string) => {
    if (!path || path === 'N/A') return <Text>N/A</Text>;
    
    const parts = path.split('/');
    const fileName = parts.pop() || '';
    const directory = parts.join('/');
    
    return (
      <div className="file-path-display">
        <FolderOutlined style={{ marginRight: '8px' }} />
        <span className="directory-path">{directory}/</span>
        <span className="file-name">{fileName}</span>
      </div>
    );
  };

  const getConfidenceBadge = (score?: number) => {
    if (score === undefined || score === null) return null;
    
    if (score <= 1) {
      return (
        <Tag color="red" style={{ marginLeft: 8 }}>
          <ExclamationCircleOutlined /> Very Low Confidence ({score}/5)
        </Tag>
      );
    } else if (score === 2) {
      return (
        <Tag color="orange" style={{ marginLeft: 8 }}>
          <ExclamationCircleOutlined /> Low Confidence ({score}/5)
        </Tag>
      );
    } else if (score === 3) {
      return (
        <Tag color="gold" style={{ marginLeft: 8 }}>
          <InfoCircleOutlined /> Moderate Confidence ({score}/5)
        </Tag>
      );
    } else if (score === 4) {
      return (
        <Tag color="lime" style={{ marginLeft: 8 }}>
          <InfoCircleOutlined /> High Confidence ({score}/5)
        </Tag>
      );
    } else {
      return (
        <Tag color="green" style={{ marginLeft: 8 }}>
          <InfoCircleOutlined /> Very High Confidence ({score}/5)
        </Tag>
      );
    }
  };

  const columnColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      width: '20%',
      render: (text: string, record: Column) => (
        record.id ? <Link to={`/columns/${record.id}`}>{text}</Link> : text
      ),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: '15%',
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      width: '55%',
      render: (_: any, record: Column) => (
        <div>
          {record.description || record.ai_description || 'No description available'}
          {record.ai_description && !record.user_edited && (
            <Tag color="purple" style={{ marginLeft: 8, fontSize: '12px', padding: '0 6px' }}>
              AI
            </Tag>
          )}
          {getConfidenceBadge(record.ai_confidence_score)}
        </div>
      ),
    },
    {
      title: 'Keys',
      key: 'keys',
      width: '10%',
      render: (_: any, record: Column) => (
        <Space>
          {record.isPrimaryKey && <Tag color="green">PK</Tag>}
          {record.isForeignKey && <Tag color="blue">FK</Tag>}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card 
        style={{ marginBottom: '16px' }}
        title={
          <Space size="large">
            <Space>
              <FileOutlined />
              <Text strong>File:</Text> 
              {formatFilePath(model.file_path || 'N/A')}
            </Space>
          </Space>
        }
        extra={
          <Space>
            <Tooltip title="Provide additional context to improve AI descriptions">
              <Button 
                icon={<InfoCircleFilled />}
                onClick={showContextModal}
                type="default"
              >
                Add Context
              </Button>
            </Tooltip>
            <Tooltip title="Refresh AI descriptions for this model and its columns (AI descriptions are also auto-generated during metadata refresh)">
              <Button 
                icon={<RobotOutlined />} 
                onClick={handleRefreshMetadata}
                loading={refreshing}
                type="primary"
              >
                Refresh AI Descriptions
              </Button>
            </Tooltip>
          </Space>
        }
      >
        <Descriptions bordered column={2}>
          <Descriptions.Item label="Project" span={1}>
            <Tag color="blue" icon={<DatabaseOutlined />}>
              {model.project}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Schema" span={1}>{model.schema || 'N/A'}</Descriptions.Item>
          <Descriptions.Item label="Materialized" span={1}>{model.materialized || 'N/A'}</Descriptions.Item>
          <Descriptions.Item label="Columns" span={1}>{model.columns?.length || 0}</Descriptions.Item>
          <Descriptions.Item label="Description" span={2}>
            {model.description || model.ai_description || 'No description available'}
            {model.ai_description && !model.user_edited && (
              <Tag color="purple" style={{ marginLeft: 8, fontSize: '12px', padding: '0 6px' }}>
                AI
              </Tag>
            )}
            {getConfidenceBadge(model.ai_confidence_score)}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Tabs defaultActiveKey="columns" type="card">
        <TabPane tab="Columns" key="columns">
          <Table 
            dataSource={model.columns || []} 
            columns={columnColumns} 
            rowKey="name" 
            pagination={{ pageSize: 20 }} 
          />
        </TabPane>
        
        <TabPane tab="SQL" key="sql">
          <pre className="sql-code">{model.sql || 'No SQL available'}</pre>
        </TabPane>
      </Tabs>

      {/* Modal for entering additional context */}
      <Modal
        title="Add Context for AI Descriptions"
        open={isContextModalVisible}
        onCancel={handleContextModalCancel}
        onOk={handleRefreshWithContext}
        okText="Refresh with Context"
        confirmLoading={refreshing}
      >
        <div style={{ marginBottom: 16 }}>
          <Text>
            Provide additional context to help the AI generate more accurate descriptions.
            This can include information about the business domain, naming conventions, 
            or specific details about how this data is used.
          </Text>
        </div>
        <TextArea
          rows={6}
          value={additionalContext}
          onChange={e => setAdditionalContext(e.target.value)}
          placeholder="For example: This model is part of our insurance claims processing system. Column prefixes like 'cust_' refer to customer data, and 'clm_' refers to claim-related fields. This data is primarily used by our claims adjusters team."
        />
      </Modal>
    </div>
  );
};

export default ModelDetailView; 