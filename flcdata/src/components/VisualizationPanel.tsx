
import {
    Stack,
    Tab,
    Tabs
} from "@mui/material";
import React, { memo } from "react";
import { PlotRenderer } from "./PlotRenderer";
import { TabContent } from "./Tabs";


type InnerProps = {
    selectedTab: string;
    data: any;
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
            <PlotRenderer data={class_per_part_hist(data)} />
            <PlotRenderer data={part_per_class_hist(data)} />
        </TabContent>
    </>
});

type VisualizationPanelProps = {
  generated: any;
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

function class_per_part_hist(data: any) {
  const n_classes = data.n_classes;
  const n_partitions = data.n_partitions;

  const X = data.X;
  const Y = data.Y;
  const PP = data.PP;


  const hist = Array.from({ length: n_partitions }, () =>
    Array(n_classes).fill(0)
  );

  for (let i = 0; i < X.length; i++) {
    const p = PP[i];
    const c = Y[i];
    hist[p][c] += 1;
  }

  const traces = hist.map((h, i) => {
    return {
      x: h,
      type: "bar",
      name: `Partition ${i}`,
      marker: {
        color: palette(i),
      },
    };
  });

  return traces;
}

function part_per_class_hist(data: any) {
  const n_classes = data.n_classes;
  const n_partitions = data.n_partitions;

  const X = data.X;
  const Y = data.Y;
  const PP = data.PP;

  const hist = Array.from({ length: n_classes }, () =>
    Array(n_partitions).fill(0)
  );

  for (let i = 0; i < X.length; i++) {
    const p = PP[i];
    const c = Y[i];
    hist[c][p] += 1;
  }

  const traces = hist.map((h, i) => {
    return {
      x: h,
      type: "bar",
      name: `Class ${i}`,
      marker: {
        color: palette(i),
      },
    };
  });

  return traces;
}
