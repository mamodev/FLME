import { VisibilityOff } from '@mui/icons-material';
import {
  Autocomplete,
  Button,
  Stack,
  TextField
} from '@mui/material';
import React from 'react';
import { useSavedFiles } from '../api/useSave';
import { useConfigContext } from '../contexts/ConfigContext';
import { useGeneratorContext } from '../contexts/GeneratorContext';
import { useSidebarContext } from '../contexts/SidebarContext';
import ModuleSelector from './ModuleSelector';



const Sidebar: React.FC = () => {
  const {
    fileSave,
    setFileSave,
    handleGenerate,
    handleSave,
    ...ctx
  } = useGeneratorContext();

  const config = useConfigContext();

  const { data: savedFiles } = useSavedFiles();
  const { hideSidebar, toggleSidebar } = useSidebarContext();

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleGenerate();
    }
  };
  

  if (hideSidebar) {
    return null;
  }



  async function handleLoad() {
    const file = savedFiles?.find((file) => file.name.split(".")[0] === fileSave);
    if (!file) {
      alert("File not found");
      return;
    }

    const params = file.generation_params;

    console.log("Loading file", file);
    console.log("Params", params);


    const dg = config.data_generators.find((dg) => dg.name === params.data_generator.name);
    const dist = config.distributions.find((dist) => dist.name === params.distribution.name);
    const part = config.partitioners.find((part) => part.name === params.partitioner.name);

    if (!dg || !dist || !part) {
      alert("Error loading file");
      return;
    }

    ctx.setDataGenerator(dg);
    ctx.setDistribution(dist);
    ctx.setPartitioner(part);

    ctx.setDataGeneratorParams(params.data_generator.parameters);
    ctx.setDistributionParams(params.distribution.parameters);
    ctx.setPartitionerParams(params.partitioner.parameters);
  }

  return (
    <Stack
      onKeyDown={handleKeyDown}
      justifyContent="flex-start"
      spacing={1}
      sx={{
        background: 'white',
        width: 300,
        borderRight: "1px solid",
        borderColor: "divider",
        padding: 2,
        maxHeight: "calc(100vh - 50px)",
        overflowY: "auto",
        boxSizing: "border-box",
      }}
    >
      <ModuleSelector category="data_generator" />
      <ModuleSelector category="distribution" />
      <ModuleSelector category="partitioner" />

      <Button
        size='small'
        variant="contained"
        onClick={handleGenerate}
      >
        Generate
      </Button>




      <Stack flex={1}  justifyContent="flex-end" spacing={1}>

      
          <Autocomplete

          value={fileSave}
      
          onInputChange={(event, newInputValue) => {
            setFileSave(newInputValue);
          }}

          fullWidth
          freeSolo
        options={(savedFiles || []).map((file) => file.name.split(".")[0])}
          renderInput={(params) => <TextField {...params} label="Saved Files" 
          />}
    
        />

          {/* <TextField
            size="small"
            label="File Save"
            value={fileSave}
            onChange={(e) => setFileSave(e.target.value)}
          /> */}

    <Stack 
          spacing={1} 
          direction="row" 
        >
          <Button
          fullWidth
          size='small'

            variant="contained"
            onClick={handleSave}
          >
            Save
          </Button>
          
          <Button fullWidth variant="contained" onClick={handleLoad}
                  size='small'

          >
            Load
          </Button>
            
        </Stack>

   
       
      </Stack>


      <Button
        variant="outlined"
        onClick={toggleSidebar}
        size='small'

        startIcon={<VisibilityOff />}
      >
        Hide Sidebar
      </Button>
    
    </Stack>
  );
};

export default Sidebar;
