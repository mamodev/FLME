import React from "react";
import {
  Dialog,
  DialogTitle,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Stack,
  Typography,
} from "@mui/material";

import { useDeleteSimulation, useSimList } from "../api/useSimList";
import { ChevronLeft, Delete, Menu, ReplayOutlined } from "@mui/icons-material";
import { SimulationMetric, useSimMetrics } from "../api/useSimMetrics";
import { PlotRenderer } from "./PlotRenderer";
import { useQueryClient } from "@tanstack/react-query";
import {
  createSlidingWindowTraces,
} from "../utils/sliding_window_trace";

export function SimulationAnalysis() {
  const queryClient = useQueryClient();

  const { data: _slist, refetch } = useSimList();
  const slist = _slist || [];
  slist.sort(
    (a, b) => a.name.localeCompare(b.name)
  )

  const [selectedSim, setSelectedSim] = React.useState<string | null>(null);
  const deleteSim = useDeleteSimulation();

  const [drawerOpen, setDrawerOpen] = React.useState(true);

  return (
    <Stack p={2} direction="row" flex={1} sx={{overflow: "hidden"}}>
      {!drawerOpen && (
        <Stack
          alignItems="center"
          justifyContent="center"
          sx={{
            borderRadius: 1,
            boxShadow: 4,
            p: 2,

            height: 10,
            width: 10,
          }}
        >
          <Menu onClick={() => setDrawerOpen(true)} />
        </Stack>
      )}

      {drawerOpen && (
        <Stack
          sx={{
            width: 300,
            borderRadius: 1,
            boxShadow: 4,
            py: 2,
            overflowY: 'auto',
            overflowX: 'hidden'
          }}
        >
          <Stack
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            px={2}
          >
            <IconButton onClick={() => setDrawerOpen(false)}>
              <ChevronLeft />
            </IconButton>

            <Typography
              variant="h6"
              fontWeight={600}
              sx={{ textDecoration: "underline" }}
            >
              Simulations
            </Typography>
            <IconButton
              onClick={() => {
                refetch();
                queryClient.invalidateQueries({ queryKey: ["sim-metrics"] });
              }}
            >
              <ReplayOutlined />
            </IconButton>
          </Stack>
          <List  dense sx={{overflow: 'auto'}}>
            {slist.map((sim, index) => (
                <ListItemButton
                  key={index}
                  onClick={() => setSelectedSim(sim.name)}
                  selected={selectedSim === sim.name}
                >
                  <IconButton
                    onClick={() => {
                      if (
                        confirm(
                          "Are you sure you want to delete this simulation?"
                        )
                      ) {
                        if (selectedSim === sim.name) {
                          setSelectedSim(null);
                        }

                        deleteSim.mutate(sim.name);
                      }
                    }}
                  >
                    <Delete />
                  </IconButton>
                  <ListItemText
                    primary={sim.name}
                    secondary={`${sim.metrics_files} / ${sim.model_files}`}
                  />
                </ListItemButton>
            ))}
          </List>
        </Stack>
      )}

      <Stack
        flex={1}
        alignItems={!!selectedSim ? undefined : "center"}
        justifyContent={!!selectedSim ? undefined : "center"}
        sx={{
          p: 2,
        }}
      >
        {selectedSim && <Simulation name={selectedSim} />}

        {!selectedSim && (
          <Typography variant="h6" textAlign="center">
            Select a simulation to see the analysis
          </Typography>
        )}
      </Stack>
    </Stack>
  );
}

function Simulation({ name }: { name: string }) {
  const { data: _metrics } = useSimMetrics(name);

  const [selectedMetric, setSelectedMetric] = React.useState<number | null>(
    null
  );

  const metrics = _metrics || [
    {
      version: 0,
      accuracy: 0,
      groups: {},
    },
  ];

  // const groups = React.useMemo(() => {
  //   return Object.keys(metrics[0].groups);
  // }, [metrics]);

  // const avg = React.useMemo(() => {
  //   const avg = Array.from({ length: metrics.length }, () => 0);

  //   for (let i = 0; i < metrics.length; i++) {
  //     const m = metrics[i];
  //     const sum = Object.values(m.groups).reduce((a, b) => a + b, 0);
  //     avg[i] = sum / groups.length;
  //   }

  //   return avg;
  // }, [metrics]);


  const data = React.useMemo(() => {
    return metrics.length == 1 ? [] : [
      ...createSlidingWindowTraces({
        x: metrics.map((m) => m.version),
        y: metrics.map((m) => m.accuracy),
        windowSize: 20,
        bandFillColor: "rgba(200,50,50,0.3)",
      }),

   
      // {
      //   x: metrics.map((m) => m.version),
      //   y: metrics.map((m) => m.accuracy),
      //   type: "scatter",
      //   mode: "lines+markers",
      //   name: "Accuracy",
      //   marker: { color: "blue" },
      // },
      // ...groups.map((group) => ({
      //   x: metrics.map((m) => m.version),
      //   y: metrics.map((m) => m.groups[group]),
      //   type: "scatter",
      //   mode: "lines+markers",
      //   visible: "legendonly",
      //   name: group,
      // })),
      // {
      //   x: metrics.map((m) => m.version),
      //   y: avg,
      //   type: "scatter",
      //   mode: "lines",
      //   name: "Average",
      //   marker: { color: "red" },
      // }
    ]
  }, [_metrics]);

  const layout = React.useMemo(() => {
    return {
        title: {
          text: `Simulation ${name.replaceAll("_", " ").replaceAll("-", " ")}`,
          font: {
            size: 20,
          },
        },
        showlegend: true,
        yaxis: {
          range: [0, 1],
          title: "Accuracy",
        },
        xaxis: {
          title: "Model version",
        },
      }}, [name]);


    
  const handlePointClick = React.useCallback(
    (point: any, trace: any) => {
      setSelectedMetric(point);
    },
    [setSelectedMetric]
  );



  if (!_metrics) {
    return <Typography variant="body1">Loading...</Typography>;
  }



  return (
    <>
      <PlotRenderer
        data={data}
        layout={layout}
        onPointClick={handlePointClick}
      />

      <Dialog
        open={!!selectedMetric}
        onClose={() => setSelectedMetric(null)}
        fullWidth
        maxWidth="lg"
      >
        {selectedMetric !== null && (
          <DetailsDialog
            metric={metrics[selectedMetric] as unknown as SimulationMetric}
          />
        
        )}
      </Dialog>
    </>
  );
}

type DetailsDialogProps = {
  metric: SimulationMetric;

}
function DetailsDialog(props: DetailsDialogProps) {
  const { metric } = props;

  const groups = Object.keys(metric.groups);
  // sort groups, ds-10 should be after ds-9
  groups.sort((a, b) => {
    const aNum = parseInt(a.split("-")[1]);
    const bNum = parseInt(b.split("-")[1]);

    if (aNum === bNum) {
      return a.localeCompare(b);
    }

    return aNum - bNum;
  });


  const gvalues = groups.map((g) => metric.groups[g]);

  return   <>
    <DialogTitle>
      Model version {metric.version} - {" "}
      {metric.accuracy}
    </DialogTitle>

    {/* PLOT HIST OF ACC PER GROUP */}
    <PlotRenderer
      data={[
        {
          x: groups,
          y: gvalues,
          type: "bar",
          name: "Groups",
        },
      ]}
    />
  </>
}
