import {
  Autocomplete,
  Button,
  Slider,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import React from "react";
import { Module } from "../../backend/interfaces";
import { generateDefaultParameters } from "../../utils/generatord";
import { ModuleRenderer } from "../ModuleRenderer";
import { TimelineGenerators } from "./generators/defs";
import { IModule, ISimulation, ITimeline, IUpdate } from "./types";
import { Timeline, TimelineProps } from "./Timeline";
import { re } from "mathjs";
import { checkEventOrder } from "./generators/check";
import { execPython, read, useLS, writeJson } from "../../api/backend";
import { computeAggregations, computeAggregationsStats } from "./strategies";
import { useDatasets, useSavedFiles } from "../../api/useSave";
import { useRunPipeline } from "../../api/pipelines";
import { ensureExtname, pj } from "../../utils/path";
import { TabContent } from "../Tabs";
import { PlotRenderer } from "../PlotRenderer";
import { PlotlyData } from "../../contexts/PlotlyCtx";
import { gaussianRandomWithParams } from "../../utils/guassian";

// function uniformPartitionDistribution(
//   nclients: number,
//   npartitions: number
// ): number[] {
//   if (nclients % npartitions !== 0) {
//     throw new Error(
//       "Number of clients must be divisible by number of partitions"
//     );
//   }

//   const clients_per_partition = Array(npartitions).fill(nclients / npartitions);

//   return clients_per_partition;
// }

function buffStrat(n: number) {
  return function (updates: IUpdate[]): boolean {
    return updates.length >= n;
  };
}

// const fedAvgUniformSimulation: ISimulation = {
//   npartitions: 8,
//   client_per_partition: uniformPartitionDistribution(8 * 10, 8),
//   naggregations: 20,
//   strategy: buffStrat,
// };

// // const timeline = generateFlSyncTimeline(fedAvgUniformSimulation, 1);

// const timeline = generateFlAsyncTimeline(
//   fedAvgUniformSimulation,
//   0.01, // fetch probability per idle client per tick
//   0.5, // train probability per fetched client per tick
//   0.5, // send probability per trained client per tick (after idle)
//   1, // min idle after train
//   3 // max idle after train
// );
// let errors = checkEventOrder(timeline);
// if (errors != null) {
//   console.log(timeline);
//   console.error(errors);
//   throw new Error(errors);
// }

const simulationModule: Module = {
  name: "Simulation",
  description: "Simulation module",
  parameters: {
    npartitions: {
      type: "int",
      default: 8,
      min: 1,
      max: 100,
    },
    client_per_partition: {
      type: "int",
      default: 10,
      min: 1,
      max: 1000,
    },

    proportionalKnowledge: {
      type: "boolean",
      default: false,
    },

    cp_distr: {
      type: "enum",
      default: "uniform",
      options: ["uniform", "random"],
    },

    naggregations: {
      type: "int",
      default: 20,
      min: 1,
      max: 1000,
    },
    nbuff: {
      type: "int",
      default: 5,
      min: 1,
      max: 1000,
    },
  },
};

export function TimelinePanel() {
  // const { data: DATASETS } = useDatasets();
  const { data: _datasets } = useLS(".data");
  const datasets = _datasets.map((d: any) => d.split("/")[1]);

  const [viewTab, setViewTab] = React.useState(0);

  const runPipeline = useRunPipeline();

  const { data: savedFiles } = useLS(".timelines");
  const [fileSave, setFileSave] = React.useState<string | null>(null);

  const [generator, _setGenerator] = React.useState<{
    mod: IModule;
    props: any;
  }>({
    mod: TimelineGenerators[0],
    props: generateDefaultParameters(TimelineGenerators[0]),
  });

  const [simulatinParams, setSimulationParams] = React.useState<any>(
    generateDefaultParameters(simulationModule)
  );

  const setGenerator = (mod: IModule) => {
    _setGenerator({
      mod,
      props: generateDefaultParameters(mod),
    });
  };

  const [generated, setGenerated] = React.useState<TimelineProps | null>(null);

  const handleGenerate = () => {

    let cpp: number[] = []
    if (simulatinParams.cp_distr === "uniform") {
      cpp = Array(simulatinParams.npartitions).fill(
        simulatinParams.client_per_partition
      );
    } else {
      cpp = Array(simulatinParams.npartitions).fill(0).map((_, i) => {
        
        const mean = simulatinParams.client_per_partition;
        const std = Math.floor(mean / 2);
          
        return Math.floor(Math.abs(gaussianRandomWithParams(mean, std)))

      })
    }


    const sim: ISimulation = {
      ...simulatinParams,
      strategy: buffStrat(simulatinParams.nbuff),
      client_per_partition: cpp,
    };

    const timeline = generator.mod.fn(sim, generator.props);

    let errors = checkEventOrder(timeline);
    if (errors != null) {
      alert(errors);
      throw new Error(errors);
    }

    setGenerated({
      sim,
      timeline,
    });

    return {
      sim,
      timeline,
    };
  };

  const [selectedDataset, setSelectedDataset] = React.useState<string | null>(
    null
  );

  const [lrRange, setLrRange] = React.useState<number[]>([0.1, 0.1]);


  async function loadFile(file: string) {
    read(".timelines/" + file)
      .then((data) => {
        const parsed = JSON.parse(data);
        const modules = parsed.modules;

        setSimulationParams(modules.simulation.parameters);

        _setGenerator({
          mod: TimelineGenerators.find(
            (mod) => mod.name === modules.generator.name
          )!,
          props: modules.generator.parameters,
        });
      })
      .catch((err) => {
        console.error(err);
        alert("Error loading file");
      });
  }

  return (
    <Stack flex={1}>
      <Stack direction="row" spacing={1} px={1}>
        <Stack
          sx={{
            border: "1px solid black",
            p: 1,
            pt: 0,
          }}
        >
          <Typography variant="h6">Simulation</Typography>

          <ModuleRenderer
            module={simulationModule}
            parameters={simulatinParams}
            onChange={(newProps: any) => {
              setSimulationParams({
                ...simulatinParams,
                ...newProps,
              });
            }}
          />
        </Stack>

        <Stack
          sx={{
            border: "1px solid black",
            p: 1,
          }}
        >
          <Autocomplete
            size="small"
            disableClearable
            value={generator.mod}
            options={TimelineGenerators}
            getOptionLabel={(option) => option.name}
            isOptionEqualToValue={(option, value) => option.name === value.name}
            onChange={(_, newValue) => {
              setGenerator(newValue);
            }}
            renderInput={(params) => (
              <TextField {...params} label="Generator" />
            )}
          />

          <ModuleRenderer
            module={generator.mod}
            parameters={generator.props}
            onChange={(newProps: any) => {
              _setGenerator({
                ...generator,
                props: {
                  ...generator.props,
                  ...newProps,
                },
              });
            }}
          />
        </Stack>

        <Stack sx={{ width: 400, pt: 1}} spacing={1}>
          <Stack direction="row" spacing={1} alignItems="center">
            <TextField
              label="Learning Rate Max"
              type="number"
              size="small"

              value={lrRange[0]}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (isNaN(val)) return;
                setLrRange([val, lrRange[1]]);
              }}
            />

            <TextField
              label="Learning Rate Min"
              size="small"
              type="number"
              value={lrRange[1]}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (isNaN(val)) return;
                setLrRange([lrRange[0], val]);
              }}
            />
          </Stack>

          <Button onClick={handleGenerate}>Generate</Button>

          <Stack direction="row" spacing={1} alignItems="center">
            <Autocomplete
              value={fileSave}
              onInputChange={(event, newInputValue) => {
                setFileSave(newInputValue);
              }}
              fullWidth
              size="small"
              freeSolo
              options={savedFiles.map((file) => file.split("/")[1])}
              renderInput={(params) => (
                <TextField {...params} label="Saved Files" />
              )}


            />

            <Button
              disabled={fileSave == null || fileSave.length === 0}
              onClick={() => {
                if (!fileSave) return;

                let gen: TimelineProps | null = generated;
                if (gen == null) gen = handleGenerate();

                const FILE = {
                  timeline: gen!.timeline,
                  sim: gen!.sim,
                  aggregations: computeAggregations(gen!.sim, gen!.timeline),
                  modules: {
                    simulation: {
                      name: simulationModule.name,
                      parameters: simulatinParams,
                    },
                    generator: {
                      name: generator.mod.name,
                      parameters: generator.props,
                    },
                  },
                };

                writeJson(
                  pj(".timelines", ensureExtname(fileSave, ".json")),
                  FILE
                )
                  .then(() => {
                    alert("Saved to " + pj(".timelines", fileSave));
                  })
                  .catch((err) => {
                    alert("Error saving file");
                  });
              }}
            >
              Save
            </Button>

            <Button
              disabled={fileSave == null || fileSave.length === 0}
              onClick={() => loadFile(fileSave!)}
            >
              Load
            </Button>
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center">
            <Autocomplete
              value={selectedDataset}
              fullWidth
              size="small"
              options={datasets}
              renderInput={(params) => (
                <TextField {...params} label="Dataset" />
              )}
              onChange={(_, newValue) => {
                setSelectedDataset(newValue);
              }}
            />

            <Button
              disabled={!fileSave && !selectedDataset}
              onClick={() => {
                if (!fileSave || !selectedDataset) return;

                let sim_name = "sim";
                if (lrRange[0] !== 0.1 || lrRange[1] !== 0.1) {
                  sim_name = `sim_(${lrRange[0]}:${lrRange[1]})`;
                }

                runPipeline
                  .mutateAsync({
                    temp_name: "timeline_pip",
                    args: {
                      SIM_NAME: sim_name,
                      DS_NAME: selectedDataset.endsWith(".npz")
                        ? selectedDataset.split(".")[0]
                        : selectedDataset,
                      TIMELINE: fileSave.endsWith(".json")
                        ? fileSave.split(".")[0]
                        : fileSave,
                      SEED: "0",
                      LR_MIN: lrRange[0].toString(),
                      LR_MAX: lrRange[1].toString(),
                    },
                  })
                  .then(() => {
                    alert("Pipeline started");
                  })
                  .catch((err) => {
                    alert("Error starting pipeline");
                  });
              }}
            >
              Run
            </Button>
          </Stack>
        </Stack>
      </Stack>

      {/* <Timeline sim={fedAvgUniformSimulation} timeline={timeline} /> */}
      <Tabs onChange={(e, newValue) => {
        setViewTab(newValue);
      }} value={viewTab} sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tab label="Timeline" />
        <Tab label="Client Contributions" />
        <Tab label="Partition Distribution" />
        <Tab label="Partition Contributions" />
      </Tabs>


      {generated != null && (
        <Stack
          sx={{
            flex: 1,
            p: 1,
          }}
        >
                
        <TabContent current={viewTab} value={0}>
          <Timeline sim={generated.sim} timeline={generated.timeline} />
        </TabContent>

        <TabContent current={viewTab} value={1}>
          <ContributionsPanel 
            sim={generated.sim}
            timeline={generated.timeline}
          />
        </TabContent>

        <TabContent current={viewTab} value={2}>
          <PartitionClientDistr sim={generated.sim} />
        </TabContent>


        <TabContent current={viewTab} value={3}>
          <PartitionContributions
            sim={generated.sim}
            timeline={generated.timeline}
          />
        </TabContent>

        </Stack>
      )}

      {generated == null && (
        <Stack
          justifyContent="center"
          alignItems="center"
        sx={{
            flex: 1,
            p: 1,
          }}
        >
          <Typography variant="h6">No timeline generated</Typography>
        </Stack>
      )}


    

     
    </Stack>
  );
}

type PartitionContributionProps = {
  sim: ISimulation;
  timeline: ITimeline;
};


function PartitionContributions(props: PartitionContributionProps) {
  const stats = React.useMemo(() => computeAggregationsStats(props.sim, props.timeline), [props.sim, props.timeline]);

  const XX = Array.from(
    { length: props.sim.npartitions },
    (_, i) => i
  );

  const YY = XX.map((p) => {
    return Object.values(stats[p]).reduce((a, b) => a + b, 0);
  });


  return (
    <Stack flex={1}>

      <PlotRenderer
        data={[
          {
            x: XX,
            y: YY,
            type: "bar",
            name: "Partition contributions",
            marker: {
              color: "rgb(100, 100, 255)",
            },
          },
        ]}
       
        layout={{
          title: "Partition Contributions",
          xaxis: {
            title: {
              text: "Partition",
            },
            tickmode: "linear",
            dtick: 1,
            tick0: 0,
          },
          yaxis: {
            title: {
              text: "Contributions",
            }
          },
        }}
      />
    </Stack>
  );
}




type PartitionClientDistr = {
  sim: ISimulation;
}

function PartitionClientDistr(props: PartitionClientDistr) {

  return <PlotRenderer
    data={[
      {
        x: Array.from({ length: props.sim.npartitions }, (_, i) => i),
        y: props.sim.client_per_partition,
        type: "bar",
        name: "Client per partition",
        marker: {
          color: "rgb(100, 100, 255)",
        },
      },
    ]}
    layout={{
      title: "Partition Client Distribution",
      xaxis: {
        title: {
          text: "Partitions",
        },
        tickmode: "linear",
        dtick: 1,
        tick0: 0,
      },
      yaxis: {
        title: {
          text: "Clients",
        },
        tickmode: "linear",
        dtick: 1,
        tick0: 0,
      },
    }}
  />

}

type ContributionsPanelProps = {
  sim: ISimulation;
  timeline: ITimeline;
};



function ContributionsPanel(props: ContributionsPanelProps) {
  

  const [range, setRange] = React.useState<number[]>([0, props.timeline.length - 1]);

  React.useEffect(() => {
    setRange([0, props.timeline.length - 1]);
  }
  , [props.timeline]);

  const stats = React.useMemo(() => computeAggregationsStats(props.sim, props.timeline.slice(range[0], range[1])), [props.sim, props.timeline]);

  const data: PlotlyData[] = [];


  // create histogram data
  const XX = Array.from(
    { length: props.sim.npartitions },
    (_, i) => i
  );

  const maxClients = Math.max(
    ...props.sim.client_per_partition
  );


  for (let c = 0; c < maxClients; c++) {
    const yy = Array.from(
      { length: props.sim.npartitions },
      (_, i) => 0
    );

    for (let p = 0; p < props.sim.npartitions; p++) {
      if (c < props.sim.client_per_partition[p]) {
        yy[p] = stats[p][c];
      }
    }

    data.push({
      x: XX,
      y: yy,
      type: "bar",
      name: `Client ${c}`,

      marker: {
        color: `rgb(100, ${Math.floor((c / maxClients) * 255)}, 100)`,
      },

    });
  }



  return (

    <>
    <Stack flex={1}>
      
    <Slider
      value={range}
      onChange={(e, newValue) => {
        setRange(newValue as number[]);
      }}
      valueLabelDisplay="auto"
      min={0}
      max={props.timeline.length - 1}
      disableSwap
      step={1}
    />
    <PlotRenderer

    data={data}

    layout={{
      title: "Contributions",
      xaxis: {
        title: {
          text: "Partitions",
        },
        tickmode: "linear",
        dtick: 1,
        tick0: 0,
      },
      yaxis: {
        title: {
          text: "Contributions",
        }
      },
    }}

    />
    </Stack>
    </>
  );
}
