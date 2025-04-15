import { Delete } from "@mui/icons-material";
import { Button, IconButton, Stack, Typography } from "@mui/material";
import { blue } from "@mui/material/colors";
import React from "react";
import { usePipelines } from "../api/pipelines";
import { SavedFile, useData, useDeleteFile, useSavedFiles } from "../api/useSave";
import { PipelineButton } from "./PipelineButton";
import { VisualizationPanel } from "./VisualizationPanel";

export function DatasetPanel() {
  const { data: savedFiles } = useSavedFiles();

  const [selectedFile, setSelectedFile] = React.useState<SavedFile | null>(
    null
  );

  const deleteFn = useDeleteFile()


  return (
    <Stack direction="row" 
    flex={1}
 >
      <Stack sx={{ borderRight: 1, borderColor: "divider", boxSizing: "border-box", minWidth: 300 }}>
        {savedFiles?.map((file) => {
          return (
            <Stack
              key={file.name}
              direction="row"
              justifyContent="space-between"
                alignItems="center"
              sx={{
                borderBottom: 1,
                borderColor: "divider",
                px: 2,

                backgroundColor:
                    selectedFile?.name === file.name
                        ? blue[50]
                        : "transparent",


                "&:hover": {
                  backgroundColor: "rgba(51, 50, 50, 0.04)",
                  cursor: "pointer",
                },
              }}
              onClick={() => {
                setSelectedFile((prev) => {
                  if (prev?.name === file.name) {
                    return null;
                  } else {
                    return file;
                  }
                });
              }}
            >
              <Typography
                sx={{
                    fontSize: "1.2rem",
                    color: selectedFile?.name === file.name ? "primary.main" : "",
                }}
              >{file.name}</Typography>

              <IconButton size="small" onClick={(e) => {
                e.stopPropagation();

                if(confirm("Are you sure you want to delete this file?")) {
                  if (selectedFile?.name === file.name) {
                    setSelectedFile(null);
                  }

                  deleteFn.mutate(file.name)
                }

             }}>
                <Delete/>
              </IconButton>
            </Stack>
          );
        })}

        <Stack flex={1} alignItems="center" justifyContent="flex-end">
          
            <Button 
              disabled={!savedFiles?.length}
            color="error" startIcon={<Delete />} sx={{ mt: 2, mb: 2 }} onClick={() => {
                if(!savedFiles?.length) {
                  return;
                }

              setSelectedFile(null);
              if(confirm("Are you sure you want to delete all files?")) {
                for (let i = 0; i < savedFiles.length; i++) {
                  deleteFn.mutate(savedFiles[i].name)
                }
              }
            }}>
              Delete All
            </Button>
          </Stack>
      </Stack>

      {selectedFile && (
            <FileDetails file={selectedFile} />
      )}      
    </Stack>
  );
}


function getRandomInt(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

type FileDetailsProps = {
    file: SavedFile
}

function FileDetails(props: FileDetailsProps) {
    const { file } = props

    const {data: generated} = useData(file.name)

    const { data: _pipelines, refetch } = usePipelines();

    const pipelines = _pipelines?.filter((pipeline) => {
      return pipeline.dex === file.name
    }) || [];

    const running = pipelines.reduce((acc, pipeline) => {
      return acc + (pipeline.status === "RUNNING" ? 1 : 0);
    }, 0);

    const errored = pipelines.reduce((acc, pipeline) => {
      return acc + (pipeline.status === "ERROR" ? 1 : 0);
    }
    , 0);


    // interval refetch while running > 0
    React.useEffect(() => {
      if (running > 0) {
        const interval = setInterval(() => {
          refetch();
        }, 2000);

        return () => clearInterval(interval);
      }
    }, [running]);
    
    return (
        <Stack sx={{ flex: 1, p: 1}}>
            <Stack direction="row" alignItems="center" spacing={2} justifyContent="space-between">
                <Typography variant="h4">{file.name}</Typography>
           

                <Stack direction="row" spacing={1} alignItems="center">
       
                    <Typography variant="body2" color="text.secondary">
                      {pipelines.length} pipelines submitted 
                      {running ? ` (${running} running)` : ""}
                      {errored > 0 && ` (${errored} errored)`}
                    </Typography>

                    <PipelineButton variables={{
                      DATASET: file.name.split(".")[0],
                    }}
                      args={{
                        __DEX__: file.name
                      }}
                    />
                  </Stack>
            </Stack>

            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
                <Stack>
                    <Typography>Number of samples: {file.n_samples}</Typography>
                    <Typography>Number of classes: {file.n_classes}</Typography>
                    <Typography>Number of partitions: {file.n_partitions}</Typography>
                    <Typography>Number of features: {file.n_features}</Typography>
                </Stack>

                  
                       
            </Stack>

            <Stack sx={{flex:1, mt: 2, borderRadius: 1,
              border: 1,
              borderColor: "divider",
              p:1,
              boxSizing: "border-box",

            }}
              justifyContent={!generated ? "center" : undefined}
              alignItems={!generated ? "center" : undefined}
            >

            {generated && 
              <VisualizationPanel generated={generated} />
            } 

            {!generated &&
                <Typography sx={{textAlign: 'center', fontSize: '1.5rem'}}>
                    ...Loading...
                </Typography>
            }


            </Stack>

            
        </Stack>
    )

}
