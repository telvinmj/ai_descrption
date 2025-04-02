import React, { useEffect, useState, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  NodeMouseHandler,
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useNavigate } from 'react-router-dom';

interface LineageLink {
  source: string;
  target: string;
}

interface Model {
  id: string;
  name: string;
  project: string;
  description?: string | null;
  originalModels?: Model[];
  is_source?: boolean;
}

interface LineageGraphProps {
  models: Model[];
  lineage: LineageLink[];
}

const nodeWidth = 200;
const nodeHeight = 60;

const LineageGraph: React.FC<LineageGraphProps> = ({ models, lineage }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const navigate = useNavigate();

  // Create deduplicated models based on model name AND project
  const { deduplicatedModels, modelMap, deduplicatedLineage } = useMemo(() => {
    // IMPORTANT: Don't deduplicate source models or their direct targets
    // This preserves important relationships like raw_orders → stg_orders
    const crossProjectModels = new Map<string, Model[]>();
    const projectSpecificModels = new Map<string, Model[]>();
    
    // First, identify source models and their direct targets to preserve them
    const sourceModelIds = new Set<string>();
    const targetModelIds = new Set<string>();
    
    // Find all source models (models with is_source=true) and their targets
    lineage.forEach(link => {
      const sourceModel = models.find(m => m.id === link.source);
      const targetModel = models.find(m => m.id === link.target);
      
      if (sourceModel && sourceModel.is_source) {
        sourceModelIds.add(sourceModel.id);
        if (targetModel) {
          targetModelIds.add(targetModel.id);
        }
      }
    });
    
    // Separate models that should be deduplicated across projects from project-specific models
    models.forEach(model => {
      if (!model.name) return;
      
      // Never deduplicate source models or their direct targets
      const isSourceOrTarget = sourceModelIds.has(model.id) || targetModelIds.has(model.id);
      
      // Check if this is a cross-project model (starts with fct_, stg_, or dim_)
      // but only if it's not a source model or direct target of a source
      const isCrossProject = !isSourceOrTarget && 
        (model.name.startsWith('fct_') || model.name.startsWith('stg_') || model.name.startsWith('dim_'));
      
      // For source models and their targets, include project in the key to keep them separate
      const key = isSourceOrTarget || !isCrossProject 
        ? `${model.name}_${model.project}` 
        : model.name;
      
      const targetMap = isCrossProject ? crossProjectModels : projectSpecificModels;
      
      if (!targetMap.has(key)) {
        targetMap.set(key, []);
      }
      
      targetMap.get(key)?.push(model);
    });
    
    // Log for debugging
    console.log("Cross-project models:", Array.from(crossProjectModels.keys()));
    console.log("Project-specific models:", Array.from(projectSpecificModels.keys()));
    console.log("Source models:", Array.from(sourceModelIds));
    console.log("Target models:", Array.from(targetModelIds));
    
    // Create deduplicated models list
    const dedupedModels: Model[] = [];
    const modelIdMap: Record<string, string> = {}; // Maps original IDs to new IDs
    let dedupeIndex = 0;
    
    // Process cross-project models - deduplicate by name only (ignoring project)
    // For models that appear in multiple projects, prefer the "home" project version
    crossProjectModels.forEach((modelsWithSameName, modelName) => {
      const deduplicatedId = `dedupe_${dedupeIndex++}`;
      
      // Try to find the "home" project version - the model with the same project
      // as is indicated in the model name (e.g., stg_customer in customer_project)
      const modelNameParts = modelName.split('_');
      const entityName = modelNameParts.length > 1 ? modelNameParts[1] : '';
      
      // Find the model from the project that "owns" this entity, if it exists
      const homeProject = modelsWithSameName.find(model => 
        model.project && model.project.toLowerCase().includes(entityName.toLowerCase())
      );
      
      // Use the home project model if found, otherwise use the first one
      const baseModel = homeProject || modelsWithSameName[0];
      
      // Keep the original project info to maintain correct visual grouping
      const consolidatedModel: Model = {
        ...baseModel,
        id: deduplicatedId,
        name: modelName,
        // Store original models for reference
        originalModels: modelsWithSameName
      };
      
      dedupedModels.push(consolidatedModel);
      
      // Map all original model IDs to the new consolidated ID
      modelsWithSameName.forEach(model => {
        if (model.id) {
          modelIdMap[model.id] = deduplicatedId;
        }
      });
    });
    
    // Process project-specific models - deduplicate by name AND project
    projectSpecificModels.forEach((modelsWithNameAndProject, key) => {
      const deduplicatedId = `dedupe_${dedupeIndex++}`;
      const baseModel = modelsWithNameAndProject[0];
      
      const consolidatedModel: Model = {
        ...baseModel,
        id: deduplicatedId,
        // Store original models for reference
        originalModels: modelsWithNameAndProject
      };
      
      dedupedModels.push(consolidatedModel);
      
      // Map all original model IDs to the new consolidated ID
      modelsWithNameAndProject.forEach(model => {
        if (model.id) {
          modelIdMap[model.id] = deduplicatedId;
        }
      });
    });
    
    // Log the mapping for debugging
    console.log("Model ID mapping:", modelIdMap);
    console.log("Input lineage:", lineage);
    
    // Deduplicate lineage connections
    const seenConnections = new Set<string>();
    const dedupedLineage: LineageLink[] = [];
    
    lineage.forEach(link => {
      if (!link.source || !link.target) return;
      
      const sourceId = modelIdMap[link.source];
      const targetId = modelIdMap[link.target];
      
      if (!sourceId || !targetId) {
        console.log(`Missing mapping for source: ${link.source} or target: ${link.target}`);
        return;
      }
      
      // Skip if sourceId and targetId are the same (self-reference after deduplication)
      if (sourceId === targetId) return;
      
      const connectionKey = `${sourceId}-${targetId}`;
      
      // Only add if we haven't seen this connection before
      if (!seenConnections.has(connectionKey)) {
        seenConnections.add(connectionKey);
        dedupedLineage.push({
          source: sourceId,
          target: targetId
        });
      }
    });
    
    console.log("Output lineage:", dedupedLineage);
    
    return { 
      deduplicatedModels: dedupedModels, 
      modelMap: modelIdMap,
      deduplicatedLineage: dedupedLineage
    };
  }, [models, lineage]);

  // Function to highlight a model and its connections
  const highlightModelConnections = (modelId: string | null) => {
    if (!modelId) {
      // Reset all highlights if no model is selected
      setNodes((nds) =>
        nds.map((node) => ({
          ...node,
          style: {
            ...node.style,
            background: '#fff',
            border: '1px solid #ddd',
            boxShadow: 'none',
          },
        }))
      );
      
      setEdges((eds) =>
        eds.map((edge) => ({
          ...edge,
          style: { stroke: '#aaa' },
          animated: false,
        }))
      );
      return;
    }

    // Find all connected edges (upstream and downstream)
    const connectedEdges = deduplicatedLineage.filter(
      (link) => link.source === modelId || link.target === modelId
    );
    
    const connectedNodeIds = new Set<string>();
    connectedNodeIds.add(modelId); // Add the selected node itself
    
    // Add all nodes connected by edges
    connectedEdges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    // Update node styles
    setNodes((nds) =>
      nds.map((node) => {
        const isSelected = node.id === modelId;
        const isConnected = connectedNodeIds.has(node.id);
        
        return {
          ...node,
          style: {
            ...node.style,
            background: isSelected ? '#e6f7ff' : isConnected ? '#f0f9ff' : '#fff',
            border: isSelected ? '2px solid #1890ff' : isConnected ? '1px solid #69c0ff' : '1px solid #ddd',
            boxShadow: isSelected ? '0 0 10px #1890ff' : isConnected ? '0 0 5px rgba(24, 144, 255, 0.3)' : 'none',
            opacity: isConnected || isSelected ? 1 : 0.5,
          },
        };
      })
    );

    // Update edge styles
    setEdges((eds) =>
      eds.map((edge) => {
        const isConnected = 
          edge.source === modelId || 
          edge.target === modelId;
        
        return {
          ...edge,
          style: { 
            stroke: isConnected ? '#1890ff' : '#aaa',
            strokeWidth: isConnected ? 2 : 1,
          },
          animated: isConnected,
          opacity: isConnected ? 1 : 0.3,
        };
      })
    );
  };

  // Handle node click
  const onNodeClick: NodeMouseHandler = (_, node) => {
    // Find the original model for this deduplicated node
    const deduplicatedModel = deduplicatedModels.find(model => model.id === node.id);
    if (deduplicatedModel?.originalModels && deduplicatedModel.originalModels.length > 0) {
      // Navigate directly to the model detail page without showing any popups
      const originalModelId = deduplicatedModel.originalModels[0].id;
      navigate(`/models/${originalModelId}`);
    }
  };

  // Handle node hover to highlight connections
  const onNodeMouseEnter: NodeMouseHandler = (_, node) => {
    setSelectedModelId(node.id);
    highlightModelConnections(node.id);
  };

  const onNodeMouseLeave: NodeMouseHandler = () => {
    setSelectedModelId(null);
    highlightModelConnections(null);
  };

  // Click on the background to clear selection
  const onPaneClick = () => {
    setSelectedModelId(null);
    highlightModelConnections(null);
  };

  useEffect(() => {
    if (!deduplicatedModels || deduplicatedModels.length === 0) {
      console.log("No deduplicated models available");
      return;
    }
    
    // Create nodes from the models
    const flowNodes: Node[] = [];
    const modelLevels: Record<string, number> = {};
    
    // Process lineage to create a directed graph and calculate levels
    if (deduplicatedLineage && deduplicatedLineage.length > 0) {
      // Initialize levels for all models
      deduplicatedModels.forEach(model => {
        modelLevels[model.id] = -1; // -1 means not assigned yet
      });
      
      // Find source models (no incoming edges)
      const incomingEdges: Record<string, number> = {};
    deduplicatedLineage.forEach(link => {
        if (!incomingEdges[link.target]) {
          incomingEdges[link.target] = 0;
        }
        incomingEdges[link.target]++;
      });
      
      // Source models are those with no incoming edges
      const sourceModelIds = deduplicatedModels
        .filter(model => !incomingEdges[model.id])
        .map(model => model.id);
      
      // Assign level 0 to source models
      sourceModelIds.forEach(id => {
        modelLevels[id] = 0;
      });
      
      // Breadth-first traversal to assign levels
      let frontier = [...sourceModelIds];
    let currentLevel = 0;
      
      while (frontier.length > 0) {
        const nextFrontier: string[] = [];
        
        frontier.forEach(sourceId => {
          // Find all targets from this source
          deduplicatedLineage
            .filter(link => link.source === sourceId)
            .forEach(link => {
              // Assign level to target if it's not assigned yet or if it's a higher level
              const targetLevel = modelLevels[link.target];
              if (targetLevel === -1 || targetLevel < currentLevel + 1) {
                modelLevels[link.target] = currentLevel + 1;
                nextFrontier.push(link.target);
              }
            });
        });
        
        frontier = nextFrontier;
        currentLevel++;
      }
      
      // Assign level 0 to any unassigned models (isolated nodes)
      deduplicatedModels.forEach(model => {
        if (modelLevels[model.id] === -1) {
          modelLevels[model.id] = 0;
        }
      });
    } else {
      // If no lineage, all models are on level 0
      deduplicatedModels.forEach(model => {
        modelLevels[model.id] = 0;
      });
    }
    
    // Count models per level for positioning
    const modelsPerLevel: Record<number, number> = {};
    const modelPositionInLevel: Record<string, number> = {};
    
    // Count how many models are in each level
    Object.entries(modelLevels).forEach(([modelId, level]) => {
      if (!modelsPerLevel[level]) {
        modelsPerLevel[level] = 0;
      }
      modelPositionInLevel[modelId] = modelsPerLevel[level];
      modelsPerLevel[level]++;
    });
    
    // Create nodes with positions
    deduplicatedModels.forEach(model => {
      const level = modelLevels[model.id];
      const positionInLevel = modelPositionInLevel[model.id];
      const maxInLevel = modelsPerLevel[level];
      
      // Position nodes in a grid layout
      const xGap = Math.max(nodeWidth + 50, 250);
      const yGap = Math.max(nodeHeight + 100, 150);
      
      // Calculate x based on level and maxInLevel to center nodes
      const levelWidth = maxInLevel * xGap;
      const startX = -(levelWidth / 2) + (xGap / 2);
      
      const x = startX + (positionInLevel * xGap);
      const y = level * yGap;
      
      flowNodes.push({
      id: model.id,
        position: { x, y },
      data: { 
        label: (
            <div style={{ padding: '5px' }}>
              <div style={{ fontWeight: 'bold', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {model.name}
              </div>
              <div style={{ fontSize: '0.8em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {model.project}
              </div>
          </div>
          ),
          model
      },
      style: {
        background: '#fff',
        border: '1px solid #ddd',
        borderRadius: '5px',
          width: nodeWidth,
          cursor: 'pointer',
        },
      });
    });
    
    setNodes(flowNodes);
    
    // Create edges
    const flowEdges: Edge[] = deduplicatedLineage.map((link, index) => ({
      id: `${link.source}-${link.target}`,
        source: link.source,
        target: link.target,
      type: 'smoothstep',
      animated: false,
      style: { stroke: '#aaa' },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 15,
        height: 15,
        color: '#aaa',
      },
    }));
    
    setEdges(flowEdges);
  }, [deduplicatedModels, deduplicatedLineage, navigate]);

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
        onPaneClick={onPaneClick}
        fitView
        attributionPosition="bottom-right"
      >
        <Controls />
        <MiniMap />
        <Background color="#f8f8f8" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default LineageGraph; 