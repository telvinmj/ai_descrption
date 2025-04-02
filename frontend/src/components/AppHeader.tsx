import React from 'react';
import { Layout, Typography, Button, Space, Dropdown, MenuProps } from 'antd';
import { DownOutlined, DatabaseOutlined, ExportOutlined } from '@ant-design/icons';
import { exportMetadata, initializeDatabase } from '../services/api';
import { useNavigate } from 'react-router-dom';

const { Header } = Layout;
const { Title } = Typography;

const AppHeader: React.FC = () => {
  const navigate = useNavigate();

  const handleExport = async (format: 'json' | 'yaml') => {
    try {
      const data = await exportMetadata(format);
      // Create a download link
      const blob = new Blob([data.content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = data.filename;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting metadata:', error);
    }
  };

  const handleInitialize = async () => {
    try {
      await initializeDatabase();
      navigate('/projects');
    } catch (error) {
      console.error('Error initializing database:', error);
    }
  };

  const exportItems: MenuProps['items'] = [
    {
      key: 'json',
      label: 'Export as JSON',
      onClick: () => handleExport('json'),
    },
    {
      key: 'yaml',
      label: 'Export as YAML',
      onClick: () => handleExport('yaml'),
    },
  ];

  return (
    <Header>
      <div className="logo">DBT Explorer</div>
      <Title level={4} className="header-title">
        DBT Unified Schema Explorer
      </Title>
      <div style={{ flex: 1 }}></div>
      <Space>
        <Button
          type="primary"
          icon={<DatabaseOutlined />}
          onClick={handleInitialize}
        >
          Initialize
        </Button>
        <Dropdown menu={{ items: exportItems }}>
          <Button icon={<ExportOutlined />}>
            Export <DownOutlined />
          </Button>
        </Dropdown>
      </Space>
    </Header>
  );
};

export default AppHeader; 