import React, { useState, useEffect } from 'react';
import { Card, Table, Spin, Alert, Typography, Tabs, Tag, Space } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import axios from 'axios';

const { Title, Text } = Typography;

// Define the shape of our model data
interface Column {
  name: string;
  type: string;
  description: string;
  isPrimaryKey: boolean;
  isForeignKey: boolean;
}

interface Model {
  id: string;
  name: string;
  project: string;
  description: string;
  columns: Column[];
  sql: string;
}

interface ModelLineage {
  source: string;
  target: string;
}

const LineageExplorer: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [lineage, setLineage] = useState<ModelLineage[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Fetch models
        const response = await axios.get<Model[]>('http://localhost:8000/api/models');
        setModels(response.data);

        // Fetch lineage relationships
        const lineageResponse = await axios.get<ModelLineage[]>('http://localhost:8000/api/lineage');
        setLineage(lineageResponse.data);

        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please try again later.');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Extract unique project names for filtering
  const uniqueProjects = [...new Set(models.map((model) => model.project))];

  const columns: ColumnsType<Model> = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
    },
    {
      title: 'Project',
      dataIndex: 'project',
      key: 'project',
      filters: uniqueProjects.map((project) => ({
        text: project,
        value: project,
      })),
      onFilter: (value, record) => record.project === value,
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (text) => <Text ellipsis={{ tooltip: text }}>{text || 'No description'}</Text>,
    },
    {
      title: 'Columns',
      key: 'columns',
      render: (_, record) => <span>{record.columns ? record.columns.length : 0} columns</span>,
    },
    {
      title: 'Dependencies',
      key: 'dependencies',
      render: (_, record) => {
        const dependencies = lineage.filter((item) => item.target === record.id);
        return <span>{dependencies.length} dependencies</span>;
      },
    },
  ];

  if (loading) {
    return <Spin size="large" tip="Loading models..." />;
  }

  if (error) {
    return <Alert type="error" message={error} />;
  }

  return (
    <div className="lineage-explorer">
      <Title level={2}>dbt Model Explorer</Title>
      <Text type="secondary" style={{ marginBottom: 16, display: 'block' }}>
        Explore all models across your dbt projects
      </Text>

      <Card>
        <Table
          dataSource={models}
          columns={columns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          onRow={(record) => ({
            onClick: () => {
              console.log('Selected model:', record);
            },
          })}
        />
      </Card>

      {models.length > 0 && (
        <Card style={{ marginTop: 16 }}>
          <Title level={4}>Projects Summary</Title>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 16 }}>
            {uniqueProjects.map((project) => (
              <Card key={project} size="small" style={{ width: 200 }}>
                <Space direction="vertical">
                  <Tag color="blue">{project}</Tag>
                  <Text>
                    {models.filter((model) => model.project === project).length} models
                  </Text>
                </Space>
              </Card>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default LineageExplorer;


// import React, { useState, useEffect } from 'react';
// import { Card, Table, Spin, Alert, Typography, Tabs, Tag, Space } from 'antd';
// import axios from 'axios';

// const { Title, Text } = Typography;
// const { TabPane } = Tabs;

// // Define the shape of our model data
// interface Model {
//   id: string;
//   name: string;
//   project: string;
//   description: string;
//   columns: Column[];
//   sql: string;
// }

// interface Column {
//   name: string;
//   type: string;
//   description: string;
//   isPrimaryKey: boolean;
//   isForeignKey: boolean;
// }

// interface ModelLineage {
//   source: string;
//   target: string;
// }

// const LineageExplorer: React.FC = () => {
//   const [models, setModels] = useState<Model[]>([]);
//   const [lineage, setLineage] = useState<ModelLineage[]>([]);
//   const [loading, setLoading] = useState<boolean>(true);
//   const [error, setError] = useState<string | null>(null);
  
//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         setLoading(true);
//         // Fetch models
//         const response = await axios.get('http://localhost:8000/api/models');
//         setModels(response.data);
        
//         // Fetch lineage relationships
//         const lineageResponse = await axios.get('http://localhost:8000/api/lineage');
//         setLineage(lineageResponse.data);
        
//         setLoading(false);
//       } catch (err) {
//         console.error('Error fetching data:', err);
//         setError('Failed to load data. Please try again later.');
//         setLoading(false);
//       }
//     };
    
//     fetchData();
//   }, []);
  
//   const columns = [
//     {
//       title: 'Name',
//       dataIndex: 'name',
//       key: 'name',
//       sorter: (a: Model, b: Model) => a.name.localeCompare(b.name),
//     },
//     {
//       title: 'Project',
//       dataIndex: 'project',
//       key: 'project',
//       filters: [...new Set(models.map(model => model.project_name || model.project))].map(project => ({
//         text: project,
//         value: project,
//       })),
//       onFilter: (value: string, record: Model) => record.project === value,
//     },
//     {
//       title: 'Description',
//       dataIndex: 'description',
//       key: 'description',
//       render: (text: string) => (
//         <Text ellipsis={{ tooltip: text }}>{text || 'No description'}</Text>
//       ),
//     },
//     {
//       title: 'Columns',
//       key: 'columns',
//       render: (text: string, record: Model) => (
//         <span>{record.columns ? record.columns.length : 0} columns</span>
//       ),
//     },
//     {
//       title: 'Dependencies',
//       key: 'dependencies',
//       render: (text: string, record: Model) => {
//         const dependencies = lineage.filter(item => item.target === record.id);
//         return <span>{dependencies.length} dependencies</span>;
//       },
//     },
//   ];
  
//   if (loading) {
//     return <Spin size="large" tip="Loading models..." />;
//   }
  
//   if (error) {
//     return <Alert type="error" message={error} />;
//   }
  
//   return (
//     <div className="lineage-explorer">
//       <Title level={2}>dbt Model Explorer</Title>
//       <Text type="secondary" style={{ marginBottom: 16, display: 'block' }}>
//         Explore all models across your dbt projects
//       </Text>
      
//       <Card>
//         <Table 
//           dataSource={models}
//           columns={columns}
//           rowKey="id"
//           pagination={{ pageSize: 10 }}
//           onRow={(record) => ({
//             onClick: () => {
//               // Handle row click - future implementation could navigate to model detail
//               console.log('Selected model:', record);
//             },
//           })}
//         />
//       </Card>
      
//       {models.length > 0 && (
//         <Card style={{ marginTop: 16 }}>
//           <Title level={4}>Projects Summary</Title>
//           <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 16 }}>
//             {[...new Set(models.map(model => model.project))].map(project => (
//               <Card key={project} size="small" style={{ width: 200 }}>
//                 <Space direction="vertical">
//                   <Tag color="blue">{project}</Tag>
//                   <Text>
//                     {models.filter(model => model.project === project).length} models
//                   </Text>
//                 </Space>
//               </Card>
//             ))}
//           </div>
//         </Card>
//       )}
//     </div>
//   );
// };

// export default LineageExplorer; 