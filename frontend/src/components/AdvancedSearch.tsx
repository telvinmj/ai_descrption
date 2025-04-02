import React, { useState, useEffect, useRef } from 'react';
import { Input, Select, Button, Form, Card, Tag, Space, Tooltip } from 'antd';
import { SearchOutlined, FilterOutlined, ClearOutlined } from '@ant-design/icons';
import { getProjects } from '../services/api';
import { Project } from '../types';

const { Option } = Select;

interface AdvancedSearchProps {
  onSearch: (filters: {
    projectId?: string;
    search?: string;
    tag?: string;
    materialized?: string;
  }) => void;
  initialFilters?: {
    projectId?: string;
    search?: string;
    tag?: string;
    materialized?: string;
  };
}

const AdvancedSearch: React.FC<AdvancedSearchProps> = ({ 
  onSearch, 
  initialFilters = {} 
}) => {
  const [form] = Form.useForm();
  const [projects, setProjects] = useState<Project[]>([]);
  const [expandedSearch, setExpandedSearch] = useState<boolean>(false);
  const [activeFilters, setActiveFilters] = useState<string[]>([]);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Initialize form with any provided initial values
    form.setFieldsValue({
      projectId: initialFilters.projectId || undefined,
      search: initialFilters.search || '',
      tag: initialFilters.tag || undefined,
      materialized: initialFilters.materialized || undefined
    });

    // Update active filters display
    const filters = [];
    if (initialFilters.projectId) filters.push('Project');
    if (initialFilters.search) filters.push('Text');
    if (initialFilters.tag) filters.push('Tag');
    if (initialFilters.materialized) filters.push('Materialized');
    setActiveFilters(filters);

    // Fetch projects for the project filter dropdown
    fetchProjects();
    
    // Cleanup timeout on unmount
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [initialFilters, form]);

  const fetchProjects = async () => {
    try {
      const projectsData = await getProjects();
      setProjects(projectsData);
    } catch (error) {
      console.error('Error fetching projects:', error);
    }
  };

  const handleSearch = () => {
    const values = form.getFieldsValue();
    
    // Update active filters display
    const filters = [];
    if (values.projectId) filters.push('Project');
    if (values.search) filters.push('Text');
    if (values.tag) filters.push('Tag');
    if (values.materialized) filters.push('Materialized');
    setActiveFilters(filters);
    
    onSearch(values);
  };

  const handleClear = () => {
    form.resetFields();
    setActiveFilters([]);
    onSearch({});
  };

  const handleQuickSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const search = e.target.value.trim();
    form.setFieldValue('search', search);
    
    // Only update activeFilters if search has content
    const newFilters = [...activeFilters];
    const textIndex = newFilters.indexOf('Text');
    
    if (search && textIndex === -1) {
      newFilters.push('Text');
    } else if (!search && textIndex !== -1) {
      newFilters.splice(textIndex, 1);
    }
    
    setActiveFilters(newFilters);
    
    // Clear any existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    // Apply search filter immediately without debounce
    console.log('Performing immediate search for models containing:', search);
    
    // Apply search filter immediately
    onSearch({
      ...form.getFieldsValue(),
      search
    });
  };

  const toggleExpand = () => {
    setExpandedSearch(!expandedSearch);
  };

  // Common materialization types
  const materializationTypes = [
    'view',
    'table',
    'incremental',
    'ephemeral',
    'snapshot'
  ];

  // Common tags found in dbt projects
  const commonTags = [
    'daily',
    'weekly',
    'monthly',
    'reporting',
    'staging',
    'intermediate',
    'mart',
    'core',
    'curated'
  ];

  return (
    <div className="advanced-search">
      <Card 
        title={
          <div className="search-card-title">
            <span>Search Models</span>
            {activeFilters.length > 0 && (
              <Space size={[0, 8]} wrap style={{ marginLeft: 12 }}>
                {activeFilters.map(filter => (
                  <Tag key={filter} color="blue">{filter}</Tag>
                ))}
              </Space>
            )}
          </div>
        }
        extra={
          <Button 
            type="text" 
            icon={<FilterOutlined />} 
            onClick={toggleExpand}
          >
            {expandedSearch ? 'Simple Search' : 'Advanced Search'}
          </Button>
        }
      >
        <Form form={form} layout="vertical" onFinish={handleSearch}>
          {/* Quick Search Input - Always Visible */}
          <Form.Item name="search" style={{ marginBottom: expandedSearch ? 24 : 0 }}>
            <Tooltip title="Search for models with names containing your search term">
              <Input 
                placeholder="Type model name to search..." 
                prefix={<SearchOutlined />}
                onChange={handleQuickSearch}
                allowClear
                addonAfter={<span style={{ fontSize: '11px', color: '#888' }}>Model Name</span>}
              />
            </Tooltip>
          </Form.Item>

          {/* Advanced Search Options - Expandable */}
          {expandedSearch && (
            <>
              <Form.Item name="projectId" label="Project">
                <Select 
                  placeholder="Select project" 
                  allowClear
                  style={{ width: '100%' }}
                >
                  {projects.map(project => (
                    <Option key={project.id} value={project.id}>{project.name}</Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item name="tag" label="Tag">
                <Select 
                  placeholder="Select tag" 
                  allowClear
                  style={{ width: '100%' }}
                >
                  {commonTags.map(tag => (
                    <Option key={tag} value={tag}>{tag}</Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item name="materialized" label="Materialization">
                <Select 
                  placeholder="Select materialization" 
                  allowClear
                  style={{ width: '100%' }}
                >
                  {materializationTypes.map(type => (
                    <Option key={type} value={type}>{type}</Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<SearchOutlined />} 
                    onClick={handleSearch}
                  >
                    Search
                  </Button>
                  <Button 
                    icon={<ClearOutlined />} 
                    onClick={handleClear}
                  >
                    Clear Filters
                  </Button>
                </Space>
              </Form.Item>
            </>
          )}
        </Form>
      </Card>
    </div>
  );
};

export default AdvancedSearch; 