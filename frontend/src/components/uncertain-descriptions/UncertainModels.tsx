import React, { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Table, Tag, Typography, Spin, Alert, Space } from 'antd';
import { ExclamationCircleOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { getUncertainModels } from '../../services/api';

const { Title, Text } = Typography;

interface UncertainModel {
  id: string;
  name: string;
  project: string;
  description: string;
  ai_confidence_score: number;
}

export const UncertainModels: React.FC = () => {
  const [models, setModels] = useState<UncertainModel[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadUncertainModels = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getUncertainModels();
      console.log('Uncertain models API response:', data);
      setModels(data.models || []);
      console.log('Models array after setting:', data.models || []);
      setError(null);
    } catch (err) {
      console.error('Error loading uncertain models:', err);
      setError('Failed to load uncertain models. Please try again later.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUncertainModels();
  }, [loadUncertainModels]);

  const getConfidenceBadge = (score: number) => {
    if (score <= 1) {
      return <Tag color="red"><ExclamationCircleOutlined /> Very Low Confidence</Tag>;
    } else if (score === 2) {
      return <Tag color="orange"><ExclamationCircleOutlined /> Low Confidence</Tag>;
    } else if (score === 3) {
      return <Tag color="gold"><InfoCircleOutlined /> Moderate Confidence (Needs Review)</Tag>;
    } else if (score === 4) {
      return <Tag color="lime"><InfoCircleOutlined /> High Confidence</Tag>;
    } else {
      return <Tag color="green"><InfoCircleOutlined /> Very High Confidence</Tag>;
    }
  };

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: UncertainModel) => (
        <Link to={`/models/${record.id}`}>{text}</Link>
      ),
    },
    {
      title: 'Project',
      dataIndex: 'project',
      key: 'project',
      render: (text: string) => (
        <Tag color="blue">{text}</Tag>
      ),
    },
    {
      title: 'Confidence',
      dataIndex: 'ai_confidence_score',
      key: 'ai_confidence_score',
      render: (score: number) => getConfidenceBadge(score),
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (text: string) => (
        text ? 
          <div className="description-cell">{text}</div> : 
          <Text type="secondary">No description available</Text>
      ),
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return <Alert type="error" message={error} />;
  }

  return (
    <div className="uncertain-models-section">
      <Space direction="vertical" style={{ width: '100%' }}>
        <Title level={3}>Models with Low Confidence Descriptions</Title>
        <Text type="secondary">
          These models have AI-generated descriptions with low confidence scores (3 or below) and may benefit from human review.
        </Text>
        
        {models.length === 0 ? (
          <Alert
            type="success"
            message="No Uncertain Models"
            description="All models have descriptions with acceptable confidence levels."
            showIcon
          />
        ) : (
          <Table
            dataSource={models}
            columns={columns}
            rowKey="id"
            pagination={{ pageSize: 10 }}
          />
        )}
      </Space>
    </div>
  );
}; 