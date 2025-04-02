import React from 'react';
import { Typography, Row, Col, Card, Statistic, Button } from 'antd';
import { 
  DatabaseOutlined, 
  TableOutlined, 
  ColumnWidthOutlined, 
  ApartmentOutlined 
} from '@ant-design/icons';
import { Link } from 'react-router-dom';

const { Title } = Typography;

const Dashboard: React.FC = () => {
  return (
    <div>
      <Title level={2}>Dashboard</Title>
      
      <Row gutter={[16, 16]} className="dashboard-stats">
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <DatabaseOutlined />
            <Statistic title="Projects" value={3} />
            <Button type="link" size="small">
              <Link to="/projects">View All</Link>
            </Button>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <TableOutlined />
            <Statistic title="Models" value={9} />
            <Button type="link" size="small">
              <Link to="/models">View All</Link>
            </Button>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <ColumnWidthOutlined />
            <Statistic title="Columns" value={28} />
            <Button type="link" size="small">
              <Link to="/columns">View All</Link>
            </Button>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card className="stat-card">
            <ApartmentOutlined />
            <Statistic title="Lineage Connections" value={8} />
            <Button type="link" size="small">
              <Link to="/lineage">Explore</Link>
            </Button>
          </Card>
        </Col>
      </Row>
      
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Welcome to dbt Metadata Explorer">
            <p>
              This application provides a comprehensive view of your dbt projects, models, and their relationships.
              Use the navigation menu on the left to explore your projects, models, columns, and lineage.
            </p>
            <p>
              <strong>Features:</strong>
            </p>
            <ul>
              <li>View and explore data across multiple dbt projects</li>
              <li>Discover model relationships and lineage</li>
              <li>Get AI-powered suggestions for model improvements</li>
              <li>Edit model and column descriptions</li>
            </ul>
            <Button type="primary">
              <Link to="/lineage">Explore Lineage</Link>
            </Button>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard; 