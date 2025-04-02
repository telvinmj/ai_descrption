import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Spin, Alert, Layout, Menu, Space, Button, Tag, Table } from 'antd';
import { 
  TableOutlined, 
  ApartmentOutlined, 
  DashboardOutlined,
  DatabaseOutlined,
  SyncOutlined,
  LinkOutlined,
  EditOutlined
} from '@ant-design/icons';
import { Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom';
import './App.css';
import LineageGraph from './components/LineageGraph';
import { Model, Project, LineageLink } from './types';
import { getModels, getProjects, getModelWithLineage, initializeDatabase, refreshMetadata } from './services/api';
import ModelDetail from './pages/ModelDetail';
import ExportButton from './components/ExportButton';
import WatcherStatusIndicator from './components/WatcherStatusIndicator';
import AdvancedSearch from './components/AdvancedSearch';
import { UncertainDescriptionsDashboard } from './components/uncertain-descriptions/UncertainDescriptionsDashboard';

const { Header, Content, Footer } = Layout;

// Models Table Component
interface ModelsTableProps {
  models: Model[];
  projects: Project[];
  lineage: LineageLink[];
}

const ModelsTable: React.FC<ModelsTableProps> = ({ models, projects, lineage }) => {
  const [filteredModels, setFilteredModels] = useState<Model[]>(models);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchFilters, setSearchFilters] = useState<{
    projectId?: string;
    search?: string;
    tag?: string;
    materialized?: string;
  }>({});
  const [isFiltering, setIsFiltering] = useState<boolean>(false);

  // Create a memoized version of applyFilters to avoid dependency issues
  const applyFilters = useCallback((filters: any) => {
    let results = [...models];
    
    if (filters.projectId) {
      const projectName = projects.find(p => p.id === filters.projectId)?.name;
      if (projectName) {
        results = results.filter(model => model.project === projectName);
      }
    }
    
    if (filters.search) {
      const searchLower = filters.search.toLowerCase().trim();
      // Update to search for partial matches instead of exact matches
      results = results.filter(model => 
        model.name.toLowerCase().includes(searchLower));
      
      // Log matches for debugging
      console.log(`Matches for "${searchLower}":`, results.map(m => m.name));
    }
    
    if (filters.tag) {
      results = results.filter(model => 
        model.tags && model.tags.includes(filters.tag));
    }
    
    if (filters.materialized) {
      results = results.filter(model => 
        model.materialized === filters.materialized);
    }
    
    return results;
  }, [models, projects]);

  // Update filtered models when models prop changes or when filters change
  useEffect(() => {
    // Set loading state
    setLoading(true);
    
    // Helpful log to see when filtering is being applied
    console.log('Applying filters with:', {
      hasFilters: Object.keys(searchFilters).length > 0,
      modelCount: models.length,
      searchTerm: searchFilters.search || 'none'
    });
    
    // Apply current filters or show all models
    if (
      Object.keys(searchFilters).length === 0 || 
      (!searchFilters.projectId && !searchFilters.search && !searchFilters.tag && !searchFilters.materialized)
    ) {
      setFilteredModels(models);
      setIsFiltering(false);
    } else {
      // Re-apply existing filters when models data changes
      const updated = applyFilters(searchFilters);
      setFilteredModels(updated);
      setIsFiltering(true);
    }
    
    // Reset loading state after a small delay to ensure UI updates
    setTimeout(() => {
      setLoading(false);
    }, 100);
  }, [models, searchFilters, applyFilters]);

  // Handle search/filter changes
  const handleSearch = async (filters: any) => {
    console.log('Search filters applied:', filters);
    setLoading(true);
    
    try {
      // Check if we have any actual filters
      const hasFilters = 
        filters.projectId || 
        (filters.search && filters.search.trim()) || 
        filters.tag || 
        filters.materialized;
      
      if (!hasFilters) {
        // No filters, so show all models
        console.log('No filters, showing all models:', models.length);
        setFilteredModels(models);
        setIsFiltering(false);
      } else {
        // Apply filters locally
        const localResults = applyFilters(filters);
        console.log('Filtered models:', localResults.length, 'of', models.length, 'total models');
        
        // Log the model names for debugging
        if (filters.search) {
          console.log('Models matching search term:', localResults.map(m => m.name).join(', '));
        }
        
        // Update state
        setFilteredModels(localResults);
        setIsFiltering(true);
      }
      
      // Set search filters after updating filteredModels to ensure UI reflects changes
      setSearchFilters(filters);
    } catch (error) {
      console.error('Error applying filters:', error);
      // Reset to showing all models on error
      setFilteredModels(models);
      setIsFiltering(false);
    } finally {
      setLoading(false);
    }
  };

  // Check for cross-project connections
  const hasCrossProjectConnection = (modelId: string): boolean => {
    const connections = lineage.filter(
      link => link.source === modelId || link.target === modelId
    );
    
    if (connections.length === 0) return false;
    
    // Find the model's project
    const model = models.find(m => m.id === modelId);
    if (!model) return false;
    
    // Check if any connected model is from a different project
    for (const conn of connections) {
      const connectedId = conn.source === modelId ? conn.target : conn.source;
      const connectedModel = models.find(m => m.id === connectedId);
      
      if (connectedModel && connectedModel.project !== model.project) {
        return true;
      }
    }
    
    return false;
  };

  // Define table columns
  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Model) => (
        <Space>
          <Link to={`/models/${record.id}`} className="model-link">
            {text}
          </Link>
          {hasCrossProjectConnection(record.id) && (
            <span className="cross-project-indicator">
              <LinkOutlined /> Cross-Project
            </span>
          )}
        </Space>
      ),
      sorter: (a: Model, b: Model) => a.name.localeCompare(b.name),
    },
    {
      title: 'Project',
      dataIndex: 'project',
      key: 'project',
      render: (text: string) => (
        <Tag color="blue" className="project-badge">
          {text}
        </Tag>
      ),
      filters: isFiltering ? [] : projects.map(p => ({ text: p.name, value: p.name })),
      onFilter: (value: any, record: Model) => record.project === value,
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (text: string, record: Model) => (
        <div className="description-cell">
          {text || <span className="no-description">No description</span>}
          {record.ai_description && !record.user_edited && (
            <Tag color="purple" style={{ marginLeft: 8, fontSize: '12px', padding: '0 6px' }}>
              AI
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: 'Schema',
      dataIndex: 'schema',
      key: 'schema',
      render: (text: string) => text || 'N/A',
    },
    {
      title: 'Materialized',
      dataIndex: 'materialized',
      key: 'materialized',
      render: (text: string) => (
        <Tag color={text === 'view' ? 'green' : 'orange'}>
          {text || 'N/A'}
        </Tag>
      ),
      filters: isFiltering ? [] : [
        { text: 'View', value: 'view' },
        { text: 'Table', value: 'table' },
        { text: 'Incremental', value: 'incremental' },
        { text: 'Ephemeral', value: 'ephemeral' },
      ],
      onFilter: (value: any, record: Model) => record.materialized === value,
    },
  ];

  // Render function for the table section
  const renderTableSection = () => {
    if (isFiltering && filteredModels.length === 0) {
      return (
        <div className="search-no-results">
          <Alert
            message="No models found"
            description={`No models match your search criteria "${searchFilters.search || ''}". Try a different search term.`}
            type="warning"
            showIcon
          />
        </div>
      );
    }

    return (
      <Table 
        dataSource={filteredModels} 
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={{ 
          pageSize: 15, 
          showSizeChanger: true, 
          pageSizeOptions: ['10', '15', '30', '50'],
          showTotal: (total) => `Total ${total} models` 
        }}
        key={`model-table-${filteredModels.length}-${isFiltering}`}
        id={isFiltering ? 'filtered-models-table' : 'all-models-table'}
      />
    );
  };

  return (
    <div className="models-section">
      <h2>Models</h2>
      
      <AdvancedSearch onSearch={handleSearch} initialFilters={searchFilters} />
      
      {renderTableSection()}
    </div>
  );
};

function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [lineage, setLineage] = useState<LineageLink[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [hasAIDescriptions, setHasAIDescriptions] = useState<boolean>(false);
  
  const navigate = useNavigate();
  const location = useLocation();

  // Set document title
  useEffect(() => {
    document.title = "DBT UI GENERATOR";
  }, []);

  useEffect(() => {
    console.log("App component mounted");
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Check if backend is available
        try {
          await axios.get('http://localhost:8000/api/health');
        } catch (e) {
          // If backend is not available, try to initialize the database
          try {
            await initializeDatabase();
          } catch (initError) {
            console.error("Failed to initialize database:", initError);
          }
        }
        
        // Fetch projects
        const projectsData = await getProjects();
        setProjects(projectsData);
        
        // Fetch models
        const modelsData = await getModels();
        setModels(modelsData);
        
        // Check if any models have AI descriptions
        const hasAI = modelsData.some(model => 
          model.ai_description || 
          model.columns?.some(column => column.ai_description)
        );
        setHasAIDescriptions(hasAI);
        
        // Create lineage data
        const lineageLinks: LineageLink[] = [];
        for (const model of modelsData) {
          if (model.id) {
            try {
              const modelWithLineage = await getModelWithLineage(model.id);
              
              // Add upstream links
              if (modelWithLineage.upstream) {
                for (const upstream of modelWithLineage.upstream) {
                  lineageLinks.push({
                    source: upstream.id.toString(),
                    target: model.id.toString()
                  });
                }
              }
            } catch (e) {
              console.error(`Error fetching lineage for model ${model.id}:`, e);
            }
          }
        }
        
        setLineage(lineageLinks);
        setLoading(false);
      } catch (e) {
        console.error("Error fetching data:", e);
        setError("Failed to load data from the backend. Please make sure the backend server is running.");
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Update active tab based on URL path
  useEffect(() => {
    if (location.pathname === '/') {
      setActiveTab('dashboard');
    } else if (location.pathname.startsWith('/models') && !location.pathname.includes('/models/')) {
      setActiveTab('models');
    } else if (location.pathname.startsWith('/projects')) {
      setActiveTab('projects');
    } else if (location.pathname.startsWith('/lineage')) {
      setActiveTab('lineage');
    } else if (location.pathname.startsWith('/uncertain-descriptions')) {
      setActiveTab('uncertain-descriptions');
    }
  }, [location.pathname]);

  if (loading) {
    return (
      <div className="app-loading">
        <Spin size="large" tip="Loading data from backend..." />
        <p>Please make sure the backend server is running at http://localhost:8000</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-error">
        <Alert
          type="error"
          message="Error Loading Data"
          description={
            <>
              <p>{error}</p>
              <p>
                Please ensure the backend server is running at http://localhost:8000.
                You can start it by running:
              </p>
              <pre>cd backend && python run.py</pre>
            </>
          }
        />
      </div>
    );
  }

  const handleTabChange = (key: string) => {
    setActiveTab(key);
    
    // Navigate to appropriate URL based on tab
    switch (key) {
      case 'dashboard':
        navigate('/');
        break;
      case 'models':
        navigate('/models');
        break;
      case 'lineage':
        navigate('/lineage');
        break;
      case 'uncertain-descriptions':
        navigate('/uncertain-descriptions');
        break;
    }
  };

  const renderDashboard = () => (
    <div className="dashboard-section">
      <h2>DBT UI GENERATOR</h2>
      <p>Explore your dbt projects, models, and lineage with AI-generated descriptions</p>
      
      <div className="stats-row">
        <div className="stat-card">
          <h3>Projects</h3>
          <div className="stat-value">{projects.length}</div>
        </div>
        <div className="stat-card">
          <h3>Models</h3>
          <div className="stat-value">{models.length}</div>
        </div>
        <div className="stat-card">
          <h3>Lineage Connections</h3>
          <div className="stat-value">{lineage.length}</div>
        </div>
      </div>
      
      <h3>Key Features</h3>
      <div className="feature-grid">
        <div className="feature-card">
          <h4>Dynamic Documentation</h4>
          <p>Documentation updates automatically as dbt models change</p>
        </div>
        <div className="feature-card">
          <h4>Auto-Generated AI Descriptions</h4>
          <p>Intelligent descriptions auto-generated from model code</p>
        </div>
        <div className="feature-card">
          <h4>User Corrections</h4>
          <p>Edit AI descriptions with your domain knowledge</p>
        </div>
        <div className="feature-card">
          <h4>Export Metadata</h4>
          <p>Download documentation in JSON or YAML format</p>
        </div>
      </div>
      
      <h3>Projects</h3>
      <div className="projects-grid">
        {projects.map(project => (
          <div key={project.id} className="project-card">
            <h4>{project.name}</h4>
            <p className="project-path">{project.path}</p>
            <p className="project-models">
              {models.filter(model => model.project === project.name).length} models
            </p>
          </div>
        ))}
      </div>
    </div>
  );

  const renderLineage = () => (
    <div className="lineage-section">
      <h2>Model Lineage</h2>
      <p>Visualize the relationships between your dbt models</p>
      <div className="deduplication-notice">
        <strong>Note:</strong> Models with the same name across different projects have been combined to simplify the visualization.
      </div>
      <div className="lineage-container">
        <LineageGraph models={models} lineage={lineage} />
      </div>
      <div className="lineage-legend">
        <div className="lineage-legend-item">
          <div className="lineage-legend-color legend-selected"></div>
          <span>Selected Model</span>
        </div>
        <div className="lineage-legend-item">
          <div className="lineage-legend-color legend-connected"></div>
          <span>Connected Model</span>
        </div>
        <div className="lineage-legend-item">
          <div className="lineage-legend-color legend-regular"></div>
          <span>Unrelated Model</span>
        </div>
      </div>
      <p className="lineage-hint">Click on a model to highlight its connections. Click again to reset.</p>
    </div>
  );

  return (
    <Layout className="app-layout">
      <Header className="app-header">
        <div className="header-container">
          <Link to="/" className="logo-section">
            <DatabaseOutlined className="logo-icon" />
            <span className="app-title">DBT UI GENERATOR</span>
          </Link>
          <div className="nav-section">
            <Menu
              theme="dark"
              mode="horizontal"
              selectedKeys={[activeTab]}
              onSelect={({ key }) => handleTabChange(key as string)}
            >
              <Menu.Item key="dashboard" icon={<DashboardOutlined />}>Dashboard</Menu.Item>
              <Menu.Item key="models" icon={<TableOutlined />}>Models</Menu.Item>
              <Menu.Item key="lineage" icon={<ApartmentOutlined />}>Lineage</Menu.Item>
              <Menu.Item key="uncertain-descriptions" icon={<EditOutlined />}>
                <Link to="/uncertain-descriptions">Uncertain Models</Link>
              </Menu.Item>
            </Menu>
          </div>
          <div className="action-section">
            <Space size="middle">
              <WatcherStatusIndicator />
              <Button 
                icon={<SyncOutlined />} 
                onClick={async () => {
                  try {
                    await refreshMetadata();
                    window.location.reload();
                  } catch (error) {
                    console.error('Error refreshing metadata:', error);
                  }
                }}
              >
                Refresh
              </Button>
              <ExportButton />
            </Space>
          </div>
        </div>
      </Header>
      
      <Content className="app-content">
        {hasAIDescriptions && (
          <Alert
            type="info"
            message="AI-Generated Descriptions Available"
            description="This metadata includes AI-generated descriptions for models and columns. Look for the AI badge next to descriptions. You can edit any description to improve it."
            showIcon
            closable
            style={{ marginBottom: 16 }}
          />
        )}
        
        <Routes>
          <Route path="/" element={renderDashboard()} />
          <Route path="/models" element={<ModelsTable models={models} projects={projects} lineage={lineage} />} />
          <Route path="/models/:id" element={<ModelDetail />} />
          <Route path="/lineage" element={renderLineage()} />
          <Route path="/uncertain-descriptions" element={<UncertainDescriptionsDashboard />} />
        </Routes>
      </Content>
      
      <Footer className="app-footer">
        <div className="footer-container">
          <p>DBT UI GENERATOR &copy; 2023</p>
        </div>
      </Footer>
    </Layout>
  );
}

export default App; 