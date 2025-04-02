import React from 'react';
import { Button, Dropdown, Tooltip } from 'antd';
import { DownOutlined, ExportOutlined } from '@ant-design/icons';
import { exportMetadataToJson, exportMetadataToYaml } from '../services/api';

const ExportButton: React.FC = () => {
  const items = [
    {
      key: 'json',
      label: 'Export as JSON',
      onClick: () => exportMetadataToJson()
    },
    {
      key: 'yaml',
      label: 'Export as YAML',
      onClick: () => exportMetadataToYaml()
    }
  ];

  return (
    <Tooltip title="Export metadata for all projects and models">
      <Dropdown menu={{ items }} placement="bottomRight">
        <Button icon={<ExportOutlined />}>
          Export <DownOutlined />
        </Button>
      </Dropdown>
    </Tooltip>
  );
};

export default ExportButton; 