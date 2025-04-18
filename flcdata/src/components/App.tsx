import React, { JSX } from "react";
import { Box, Stack, Tab, Tabs } from "@mui/material";
import { ConfigProvider } from "../contexts/ConfigContext";
import { GeneratorProvider } from "../contexts/GeneratorContext";
import { SidebarProvider } from "../contexts/SidebarContext";
import Sidebar from "./Sidebar";
import SidebarToggle from "./SidebarToggle";
import { VisualizationPanel } from "./VisualizationPanel";
import { TabContent } from "./Tabs";
import { SimulationAnalysis } from "./SimulationAnalysis";
import { PipelinesPanel } from "./PipelinesPanel";
import { DatasetPanel } from "./DatasetsPanel";
import { DatasetGeneratorPanel } from "./DatasetGeneratorPanel";

import { Mosaic, MosaicWindow } from 'react-mosaic-component';
import 'react-mosaic-component/react-mosaic-component.css';


const AppContent: React.FC = () => {
  const [tabIndex, setTabIndex] = React.useState(
    localStorage.getItem("tabIndex")
      ? parseInt(localStorage.getItem("tabIndex")!)
      : 0
  );

  React.useEffect(() => {
    localStorage.setItem("tabIndex", tabIndex.toString());
  }, [tabIndex]);

  return (
    <Stack
      sx={{
        boxSizing: "border-box",
        height: "100vh",
        width: "100vw",
      }}
    >
      <Tabs value={tabIndex} onChange={(_, newValue) => setTabIndex(newValue)}>
        <Tab label="Dataset generator"/>
        <Tab label="Datasets" />
        <Tab label="Simulation Analysis" />
        <Tab label="Pipelines" />
      </Tabs>

      <TabContent value={tabIndex} current={0}>
        <DatasetGeneratorPanel />
      </TabContent>
      
      <TabContent value={tabIndex} current={1}>
        <DatasetPanel />
      </TabContent>

      <TabContent value={tabIndex} current={2}>
        <SimulationAnalysis />
      </TabContent>

      <TabContent value={tabIndex} current={3}>
        <PipelinesPanel />
      </TabContent>
    </Stack>
  );
};

export type ViewId = 'a' | 'b' | 'c' | 'new';

const TITLE_MAP: Record<ViewId, string> = {
  a: 'Left Window',
  b: 'Top Right Window',
  c: 'Bottom Right Window',
  new: 'New Window',
};

const App: React.FC = () => {
  return (
    <ConfigProvider>
      <GeneratorProvider>
        <SidebarProvider>
          <AppContent />
        </SidebarProvider>
      </GeneratorProvider>
    </ConfigProvider>
  );
  // return <div id="app">
  //    <Mosaic<ViewId>
  //     renderTile={(id, path) => (
  //       <MosaicWindow<ViewId> path={path} createNode={() => 'new'} title={TITLE_MAP[id]}>
  //         <h1>{TITLE_MAP[id]}</h1>
  //       </MosaicWindow>
  //     )}
  //     initialValue={{
  //       direction: 'row',
  //       first: 'a',
  //       second: {
  //         direction: 'column',
  //         first: 'b',
  //         second: 'c',
  //       },
  //     }}
  //  />
  // </div>

};

export default App;
