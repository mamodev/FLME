import React from "react";
import { Stack, Tab, Tabs } from "@mui/material";
import { ConfigProvider } from "../contexts/ConfigContext";
import { GeneratorProvider } from "../contexts/GeneratorContext";
import { SidebarProvider } from "../contexts/SidebarContext";
import { DatasetGeneratorPanel } from "./DatasetGeneratorPanel";
import { DatasetPanel } from "./DatasetsPanel";
import { PipelinesPanel } from "./PipelinesPanel";
import { SimulationAnalysis } from "./SimulationAnalysis";
import { TabContent } from "./Tabs";
import { TimelinePanel } from "./timeline/TimelinePanel";
import { NotesPanel } from "./notes/NotesPanel";
import { PlotlyProvider } from "../contexts/PlotlyCtx";
import { DatasetHubPanel } from "./dataset-hub/DatasetHubPanel";

function AppContent() {
  const [tabIndex, setTabIndex] = React.useState(
    localStorage.getItem("tabIndex")
      ? parseInt(localStorage.getItem("tabIndex")!)
      : 0
  );

  React.useEffect(() => {
    localStorage.setItem("tabIndex", tabIndex.toString());
  }, [tabIndex]);



  const atoi = (() => {
    let count = 0;
    return () => {
      return count++;
    };
  })()

  return (
    <Stack
      sx={{
        boxSizing: "border-box",
        height: "100vh",
        width: "100vw",
        maxHeight: "100vh",
        overflow: "auto",
      }}
    >
      <Tabs value={tabIndex} onChange={(_, newValue) => setTabIndex(newValue)}>
        <Tab label="Dataset generator" />
        <Tab label="Dataset Hub" />
        <Tab label="Datasets" />
        <Tab label="Simulation Analysis" />
        <Tab label="Pipelines" />
        <Tab label="Timeline" />
        <Tab label="Notes" />
      </Tabs>

      <TabContent value={tabIndex} current={atoi()}>
        <DatasetGeneratorPanel />
      </TabContent>

      <TabContent value={tabIndex} current={atoi()}>
        <DatasetHubPanel />
      </TabContent>

      <TabContent value={tabIndex} current={atoi()}>
        <DatasetPanel />
      </TabContent>

      <TabContent value={tabIndex} current={atoi()}>
        <SimulationAnalysis />
      </TabContent>

      <TabContent value={tabIndex} current={atoi()}>
        <PipelinesPanel />
      </TabContent>

      <TabContent value={tabIndex} current={atoi()}>
        <TimelinePanel />
      </TabContent>

      <TabContent value={tabIndex} current={atoi()}>
        <NotesPanel />
      </TabContent>
    </Stack>
  );
}

function App() {
  return (
    <ConfigProvider>
      <PlotlyProvider>
        <GeneratorProvider>
          <SidebarProvider>
            <AppContent />
          </SidebarProvider>
        </GeneratorProvider>
      </PlotlyProvider>
    </ConfigProvider>
  );
}

export default App;
