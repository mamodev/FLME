import React, { useCallback } from 'react';
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  Edge,
  Position,
  Panel,
  NodeProps,
  Handle,
} from '@xyflow/react';
 
import '@xyflow/react/dist/style.css';
import { Button, IconButton, Stack, Typography } from '@mui/material';



type DataToken = "int" | "float" | "string" | "bool";
type Port = {
  "name": string;
  "type": DataToken;
}

function isUserInsertable(type: DataToken): boolean {
  return type === "int" || type === "float" || type === "string" || type === "bool";
}

type ApiNode = {
  id: string;
  inputs: Port[];
  outputs: Port[];
}

function NodeFactory(node: ApiNode) {
  function CustomNode(props: NodeProps) {

    const [inputPortTypes, setInportPortTypes] = React.useState<Record<string, "user" | "handle">>(
      () => {
        const initialTypes: Record<string, "user" | "handle"> = {};
        node.inputs.forEach((port) => {
          initialTypes[port.name] = isUserInsertable(port.type) ? "user" : "handle";
        });
        return initialTypes;
      }
    );

    return <Stack p={10} sx={{
      backgroundColor: 'white',
      border: '1px solid lightgray',
    }}>
      { node.inputs.map((port) => {
        return <Stack key={port.name} direction="row" spacing={1}>
          <Typography>
            {port.name}
          </Typography>
          { inputPortTypes[port.name] === "user" ?
          <Stack direction="row" spacing={1} position={"relative"}>
            <IconButton onClick={() => {

              setInportPortTypes((prev) => {
                const newState = { ...prev };
                newState[port.name] = "handle";
                return newState;
              }
              )} }>
              p
            </IconButton>
            <input type="text" />

            </Stack>
            :
            <Handle
              id={port.name}
              type="target"
              position={Position.Left}
              style={{ background: '#555' }}
              isConnectable={true}
            />
          }
        </Stack>
      }) }
  
      <Typography>
        Nome
      </Typography>
      
      { node.outputs.map((port) => {
        return <Stack key={port.name} direction="row" spacing={1}>
          <Typography>
            {port.name}
          </Typography>
            <Handle
              type="source"
              position={Position.Right}
              style={{ background: '#555' }}
              isConnectable={true}
            />
        </Stack>
      }) }
    </Stack>
  }

  return CustomNode;
}


const nodeTypes = {
  "custom": NodeFactory({
    id: "custom",
    inputs: [
      { name: "input1", type: "int" },
      { name: "input2", type: "float" },
      { name: "input3", type: "string" },
      { name: "input4", type: "bool" },
    ],
    outputs: [
      { name: "output1", type: "int" },
      { name: "output2", type: "float" },
      { name: "output3", type: "string" },
      { name: "output4", type: "bool" },
    ],
  })
};
 
export  function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
 
  const onConnect = useCallback(
    (conn:  Connection) => {
        const edge : Edge = {
            id: `${conn.source}-${conn.target}`,
            source: conn.source,
            target: conn.target,
            sourceHandle: conn.sourceHandle,
            targetHandle: conn.targetHandle,
            type: 'default',
            animated: true,
            data: { label: `${conn.source}-${conn.target}` },
        }
        
        setEdges((eds) => addEdge(edge, eds));
    },
    [setEdges],
  );

  const handleAddNode = useCallback(() => {
    const newNode = {
      id: `${nodes.length}`,
      type: 'custom',
      position: { x: Math.random() * 100, y: Math.random() * 100 },
      data: { label: `Node ${nodes.length}` },
    };

    setNodes((nds) => [...nds, newNode]);
  }
  , [nodes, setNodes]);

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
      >
        <Controls />
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <MiniMap />

        <Panel position="center-left">
          <Stack p={2} spacing={2} sx={{
            backgroundColor: 'white',
            border: '1px solid lightgray',
            borderRadius: '4px',
          }}>
            <Button variant="contained" onClick={handleAddNode}>
              Add Node
            </Button>

          </Stack>
        </Panel>
      </ReactFlow>
    </div>
  );
}