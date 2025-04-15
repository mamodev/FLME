import { Dialog, DialogTitle, IconButton, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Stack, TextField, Typography } from "@mui/material";
import React from "react";
import { useDeleteSimulation, useSimList } from "../api/useSimList";
import { Delete, ReplayOutlined, Science } from "@mui/icons-material";
import { useSimMetrics } from "../api/useSimMetrics";
import { PlotRenderer } from "./PlotRenderer";



export function SimulationAnalysis() {

  const  { data: _slist, refetch} = useSimList();
  const slist = _slist || [];

  const  [selectedSim, setSelectedSim] = React.useState<string | null>(null);
  const deleteSim = useDeleteSimulation();

  return (
    <Stack p={2} direction="row" flex={1}>
      <Stack
         sx={{
          width: 300,
          borderRadius: 1,
          boxShadow: 4,
          py: 2,
        }}
      >
        <Stack direction="row" alignItems="center" justifyContent="space-between" px={2}>
          <Typography variant="h6" fontWeight={600} sx={{textDecoration: "underline"}}>
            Simulations
          </Typography>
          <IconButton onClick={() => {
            refetch();
          }
          }>
            <ReplayOutlined/>
          </IconButton>
   
          </Stack>
        <List
          component="nav"
        >
          {slist.map((sim, index) => (
            <ListItem secondaryAction={
              <IconButton onClick={() => {
                if (confirm("Are you sure you want to delete this simulation?")) {
                  if (selectedSim === sim.name) {
                    setSelectedSim(null);
                  }
  
                  deleteSim.mutate(sim.name);
                }
              }}>
                <Delete/>
              </IconButton>
            }>

            <ListItemButton key={index} onClick={() => setSelectedSim(sim.name)} selected={selectedSim === sim.name}>
              <ListItemIcon>
                <Science sx={{
                  color: selectedSim === sim.name ? "primary.main" : "text.secondary",
                }}/>
              </ListItemIcon>
              <ListItemText primary={sim.name} secondary={`${sim.metrics_files} / ${sim.model_files}`} />

            </ListItemButton>
            </ListItem>
        
          ))}
        </List>
      </Stack>

        
      <Stack 
        flex={1}
      alignItems={
          !!selectedSim ? undefined: "center"
        }
        justifyContent={
          !!selectedSim ? undefined: "center"
        }

        sx={{
          p:2,
        }}
      >

          {selectedSim && <Simulation name={selectedSim} />}

          {!selectedSim && <Typography variant="h6" textAlign="center">
            Select a simulation to see the analysis
          </Typography>}

      </Stack>
     
    </Stack>
  );
}

function Simulation({ name }: { name: string }) {
  const { data: _metrics } = useSimMetrics(name);


  const [selectedMetric, setSelectedMetric] = React.useState<number | null>(null);

  const metrics = _metrics || [
    {
      groups: {}
    }    
  ];

  const groups = React.useMemo(() => {
    return Object.keys(metrics[0].groups);
  }, [metrics]);

  const avg = React.useMemo(() => {
    const avg = Array.from({ length: metrics.length }, () => 0);

    for (let i = 0; i < metrics.length; i++) {
      const m = metrics[i];
      const sum = Object.values(m.groups).reduce((a, b) => a + b, 0);
      avg[i] = sum / groups.length;
    }
    
    return avg;
  }, [metrics]);
  

  if (!_metrics) {
    return <Typography variant="body1">Loading...</Typography>;
  }


  return <>
  <PlotRenderer
    data={[
      {
        x: metrics.map((m) => m.version),
        y: metrics.map((m) => m.accuracy),
        type: "scatter",
        mode: "lines+markers",
        name: "Accuracy",
        marker: { color: "blue" },
      },
      ...groups.map((group) => ({
        x: metrics.map((m) => m.version),
        y: metrics.map((m) => m.groups[group]),
        type: "scatter",
        mode: "lines+markers",
        visible: "legendonly",
        name: group,
      })),
      {
        x: metrics.map((m) => m.version),
        y: avg,
        type: "scatter",
        mode: "lines",
        name: "Average",
        marker: { color: "red" },
      }
    ]}

    onPointClick={(point, trace) => {
      setSelectedMetric(point)
    }}
  />

  <Dialog open={!!selectedMetric} onClose={() => setSelectedMetric(null)} fullWidth maxWidth="lg">
    {selectedMetric !== null && <>
      <DialogTitle>
        Model version {metrics[selectedMetric || 0].version} - {metrics[selectedMetric || 0].accuracy}
      </DialogTitle>

      {/* PLOT HIST OF ACC PER GROUP */}
      <PlotRenderer
        data={[
          {
            x: Object.keys(metrics[selectedMetric || 0].groups),
            y: Object.values(metrics[selectedMetric || 0].groups),
            type: "bar",
            name: "Groups",
          },
        ]}
      />
    
    </>
    }
      
  </Dialog>
    
  </>
}

