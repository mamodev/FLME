import { Add, ArrowDownward, ArrowUpward, Delete, Visibility, VisibilityOff, X } from '@mui/icons-material';
import {
  Autocomplete,
  Button,
  IconButton,
  Stack,
  TextField
} from '@mui/material';
import React from 'react';
import { useSavedFiles } from '../api/useSave';
import { useConfigContext } from '../contexts/ConfigContext';
import { useGeneratorContext } from '../contexts/GeneratorContext';
import { useSidebarContext } from '../contexts/SidebarContext';
import ModuleSelector from './ModuleSelector';
import { ModuleRenderer } from './ModuleRenderer';
import { Module } from '../backend/interfaces';
import { update } from 'plotly.js';
import { grey } from '@mui/material/colors';



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

    const transformers = params.transformers.map((t: any) => {
      const module = config.transformers.find((m) => m.name === t.name);
      if (!module) {
        throw new Error(`Module ${t.module.name} not found`);
      }

      return {
        module,
        params: t.parameters,
      };
    });

 
    ctx.setDataGenerator(dg);
    ctx.setDistribution(dist);
    ctx.setPartitioner(part);

    ctx.setDataGeneratorParams(params.data_generator.parameters);
    ctx.setDistributionParams(params.distribution.parameters);
    ctx.setPartitionerParams(params.partitioner.parameters);

    for (const transformer of transformers) {
      ctx.addTransformer(transformer.module);
      ctx.updateTransformerParams(ctx.transformers.length - 1, transformer.params);
    }

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
      <Transformers/>

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


function Transformers() {
  const ctx = useGeneratorContext();
  const {transformers} = useConfigContext();

  return <>

  {
    ctx.transformers.map((transformer, i) => {
      return <Transformer key={i} idx={i}/>
    })
  }

  <Button startIcon={<Add/>} size='small' variant="outlined"
    onClick={() => {
      ctx.addTransformer(transformers[0]);
    }}
  >
     Transformer
  </Button>
  
  </>
}

type TransformerProps = {
  idx: number;
}

function Transformer({idx}: TransformerProps) {

  const ctx = useGeneratorContext();
  const {transformers} = useConfigContext();

  const [show, setShow] = React.useState(false);

  return <Stack>

    <Stack direction="row" spacing={1} alignItems="center">

    <Stack sx={{
          "& .MuiSvgIcon-root": {
            fontSize: 18,
            cursor: "pointer",
            color: grey[500],
            "&.disabled": {
              color: grey[300],
              cursor: "not-allowed",
            },
            "&:hover": {
              color: "primary.main",
            }
          }
        }}>
          <ArrowUpward
            className={idx === 0 ? "disabled" : ""}
          onClick={() => {
            ctx.moveTransformerUp(idx);
          }}/>
          <ArrowDownward  
            className={idx === ctx.transformers.length - 1 ? "disabled" : ""}
          onClick={() => {
            ctx.moveTransformerDown(idx);
          }}/>
        </Stack>


        <IconButton size="small" onClick={() => {
          setShow(p => !p);
        }}>

          {show ? <VisibilityOff sx={{fontSize: 18}}/> : <Visibility sx={{fontSize: 18}}/>}
        </IconButton>
          
   

      <Autocomplete
      fullWidth 
          size="small"
          disableClearable
          value={ctx.transformers[idx].module}
          options={transformers}
          getOptionLabel={(option: Module) => option.name}
          renderInput={(params) => <TextField {...params} />}
          isOptionEqualToValue={(option: Module, value: Module) => 
            option.name === value.name
          }
          onChange={(_, newValue: Module | null) => {
            if (newValue) {
              ctx.setTransformerModule(idx, newValue);
              // ctx.updateTransformerParams(idx, newValue);
            }
          }}
        />

<IconButton size="small" onClick={() => {
      if (confirm("Are you sure you want to delete this transformer?")) {
        ctx.removeTransformer(idx);
      }
    }}>
      <Delete sx={{fontSize: 18}}/>
    </IconButton>

      </Stack>


    {show &&
    <ModuleRenderer module={ctx.transformers[idx].module}
      parameters={ctx.transformers[idx].params}
      onChange={(params) => {
        ctx.updateTransformerParams(idx, params);
      }}
      />
    }

  </Stack>
}

export default Sidebar;
