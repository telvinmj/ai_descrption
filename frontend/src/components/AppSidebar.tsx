import React, { useState, useEffect } from 'react';
import { Layout, Menu } from 'antd';
import type { MenuProps } from 'antd';  // Importing MenuProps
import {
  DashboardOutlined,
  DatabaseOutlined,
  TableOutlined,
  ColumnWidthOutlined,  // Corrected icon
  ApartmentOutlined
} from '@ant-design/icons';
import { Link, useLocation } from 'react-router-dom';

const { Sider } = Layout;

const AppSidebar: React.FC = () => {
  const location = useLocation();
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);

  useEffect(() => {
    const pathname = location.pathname;
    if (pathname === '/') {
      setSelectedKeys(['/']);
    } else if (pathname.startsWith('/projects')) {
      setSelectedKeys(['/projects']);
    } else if (pathname.startsWith('/models')) {
      setSelectedKeys(['/models']);
    } else if (pathname.startsWith('/columns')) {
      setSelectedKeys(['/columns']);
    } else if (pathname.startsWith('/lineage')) {
      setSelectedKeys(['/lineage']);
    }
  }, [location]);

  const items: MenuProps['items'] = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: <Link to="/">Dashboard</Link>,
    },
    {
      key: '/projects',
      icon: <DatabaseOutlined />,
      label: <Link to="/projects">Projects</Link>,
    },
    {
      key: '/models',
      icon: <TableOutlined />,
      label: <Link to="/models">Models</Link>,
    },
    {
      key: '/columns',  // Corrected missing leading slash
      icon: <ColumnWidthOutlined />,
      label: <Link to="/columns">Columns</Link>,
    },
    {
      key: '/lineage',  // Corrected missing leading slash
      icon: <ApartmentOutlined />,
      label: <Link to="/lineage">Lineage Explorer</Link>,
    },
  ];

  return (
    <Sider width={200} theme="light">
      <Menu
        mode="inline"
        selectedKeys={selectedKeys}
        style={{ height: '100%', borderRight: 0 }}
        items={items}
      />
    </Sider>
  );
};

export default AppSidebar;


// import React, { useState, useEffect } from 'react';
// import { Layout, Menu } from 'antd';
// import {
//   DashboardOutlined,
//   DatabaseOutlined,
//   TableOutlined,
//   ColumnWidthOutlined,
//   ApartmentOutlined
// } from '@ant-design/icons';
// import { Link, useLocation } from 'react-router-dom';

// const { Sider } = Layout;

// const AppSidebar: React.FC = () => {
//   const location = useLocation();
//   const [selectedKeys, setSelectedKeys] = useState<string[]>([]);

//   useEffect(() => {
//     const pathname = location.pathname;
//     if (pathname === '/') {
//       setSelectedKeys(['dashboard']);
//     } else if (pathname.startsWith('/projects')) {
//       setSelectedKeys(['projects']);
//     } else if (pathname.startsWith('/models')) {
//       setSelectedKeys(['models']);
//     } else if (pathname.startsWith('/columns')) {
//       setSelectedKeys(['columns']);
//     } else if (pathname.startsWith('/lineage')) {
//       setSelectedKeys(['lineage']);
//     }
//   }, [location]);

//   const items: MenuProps['items'] = [
//     {
//       key: '/',
//       icon: <DashboardOutlined />,
//       label: <Link to="/">Dashboard</Link>,
//     },
//     {
//       key: '/projects',
//       icon: <DatabaseOutlined />,
//       label: <Link to="/projects">Projects</Link>,
//     },
//     {
//       key: '/models',
//       icon: <TableOutlined />,
//       label: <Link to="/models">Models</Link>,
//     },
//     {
//       key: 'columns',
//       icon: <ColumnHeightOutlined />,
//       label: <Link to="/columns">Columns</Link>,
//     },
//     {
//       key: 'lineage',
//       icon: <ApartmentOutlined />,
//       label: <Link to="/lineage">Lineage Explorer</Link>,
//     },
//   ];

//   return (
//     <Sider width={200} theme="light">
//       <Menu
//         mode="inline"
//         selectedKeys={selectedKeys}
//         style={{ height: '100%', borderRight: 0 }}
//         items={items}
//       />
//     </Sider>
//   );
// };

// export default AppSidebar; 