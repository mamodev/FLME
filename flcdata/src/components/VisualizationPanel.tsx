
import {
    Stack,
    Tab,
    Tabs
} from "@mui/material";

import React, { memo } from "react";
import { PlotRenderer } from "./PlotRenderer";
import { TabContent } from "./Tabs";
import { GenerateResponse } from "../api/useGenerate";


type InnerProps = {
    selectedTab: string;
    data: GenerateResponse
};


const Inner = memo(function (props: InnerProps) {
    const { selectedTab, data } = props;

    return <>
            <TabContent current={selectedTab} value={"3D Scatter"}>
            <PlotRenderer
            data={Array.from({ length: data.n_classes }, (_, c) => {
                const X = data.X.filter((d, i) => data.Y[i] === c);

                return {
                x: X.map((d) => d[0]),
                y: X.map((d) => d[1]),
                z: X.map((d) => d[2]),
                mode: "markers",
                type: "scatter3d",
                marker: {
                    color: palette(c),
                    size: 5,
                },
                };
            })}
            />

            <PlotRenderer
            data={Array.from({ length: data.n_partitions }, (_, p) => {
                const X = data.X.filter((d, i) => data.PP[i] === p);
                return {
                x: X.map((d) => d[0]),
                y: X.map((d) => d[1]),
                z: X.map((d) => d[2]),
                mode: "markers",
                type: "scatter3d",
                marker: {
                    color: palette(p),
                    size: 5,
                },
                };
            })}
            />
        </TabContent>

        <TabContent current={selectedTab} value={"Distribution"}>
            <PlotRenderer data={class_per_part_hist(data)} layout={ {
                barmode: 'group', 
                yaxis: {
                  title: 'Partition counts',
                  side: 'left'
                },
                yaxis2: {
                  title: 'Total per class',
                  overlaying: 'y',        
                  side: 'right'
                }
              }}/>
              
            <PlotRenderer data={part_per_class_hist(data)} 
              layout={{
                barmode: 'group', 
                yaxis: {
                  title: 'Partition counts',
                  side: 'left'
                },
                yaxis2: {
                  title: 'Total per class',
                  overlaying: 'y',        
                  side: 'right'
                }
              }}
            />
        </TabContent>
    </>
});

type VisualizationPanelProps = {
  generated: GenerateResponse | null;
}

export function VisualizationPanel(props: VisualizationPanelProps) {
  const { generated } = props;
  

  const tabs = ["3D Scatter", "Distribution", "Partitioning"];
  const [selectedTab, _setSelectedTab] = React.useState<string>(() => {
    const storedTab = localStorage.getItem("selectedTab");
    if (storedTab && tabs.includes(storedTab)) {
      return storedTab;
    }
    return tabs[0];
  });

  const setSelectedTab = (newValue: string) => {
    localStorage.setItem("selectedTab", newValue);
    _setSelectedTab(newValue);
  };

  return (
    <Stack
      flex={1}
      sx={{
        background: "white",
        boxSizing: "border-box",
      }}
    >
      <Tabs
        value={selectedTab}
        onChange={(e, newValue) => {
          console.log("selected tab", newValue);
          setSelectedTab(newValue);
        }}
      >
        {tabs.map((tab) => {
          return <Tab label={tab} key={tab} value={tab} />;
        })}
      </Tabs>

      {!!generated && 
        <Inner selectedTab={selectedTab} data={generated} />
     }

     
    </Stack>
  );
}

function palette(n: number) {
  const colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];

  return colors[n % colors.length];
}

function part_per_class_hist(data: GenerateResponse) {
  const n_classes = data.n_classes;
  const n_partitions = data.n_partitions;


  const hist = Array.from({ length: n_partitions }, (_, p) =>
    Array.from({ length: n_classes }, (_, c) => 
      data.CTP[c][p] 
    ) 
  );


  const traces : Plotly.Data[] = [];

  const xx = Array.from({ length: n_classes }, (_, c) => c);

  for (let p = 0; p < n_partitions; p++) {
    traces.push({
      x: xx,
      y: hist[p],
      type: "bar",
      name: `Partition ${p}`,
      yaxis: "y2",
      marker: {
        color: palette(p) 
      },
    });
  }

  for (let c = 0; c < n_classes; c++) {
    traces.push({
      x: [c],
      y: [hist.map(h => h[c]).reduce((sum, count) => sum + count, 0)],
      type: "bar",
      name: `Total per class ${c}`,
      yaxis: "y",
      marker: {
        color: palette(c) + "50",
      },
    });
  }

    
  return traces;
}

function class_per_part_hist(data: GenerateResponse) {
  const n_classes = data.n_classes;
  const n_partitions = data.n_partitions;

  const hist = Array.from({ length: n_classes }, (_, c) =>
    Array.from({ length: n_partitions }, (_, p) => 
      data.CTP[c][p] 
    ) 
  );

  const traces : Plotly.Data[] = [];

  const xx = Array.from({ length: n_partitions }, (_, p) => p);

  for (let p = 0; p < n_classes; p++) {
    traces.push({
      x: xx,
      y: hist[p],
      type: "bar",
      name: `Class ${p}`,
      yaxis: "y2",
      marker: {
        color: palette(p) 
      },
    });
  }

  for (let c = 0; c < n_partitions; c++) {
    traces.push({
      x: [c],
      y: [hist.map(h => h[c]).reduce((sum, count) => sum + count, 0)],
      type: "bar",
      name: `Total per partition ${c}`,
      yaxis: "y",
      marker: {
        color: palette(c) + "50",
      },
    });
  }


  return traces;
}
