import { Autocomplete, Button, ButtonGroup, Dialog, Stack, TextField, Typography } from "@mui/material";
import { PipelineTemplate, usePipelineTemplates, useRunPipeline } from "../api/pipelines";
import React from "react";
import { Settings, StartOutlined } from "@mui/icons-material";


// function processArg(args: string, ds_name: string) {
//     // replace __DATASET__ with the dataset name
//     // replace __IRAND(MIN, MAX) with a random integer between MIN and MAX
//     // replace __FRAND(MIN, MAX) with a random float between MIN and MAX

//     const regex = /__IRAND\((\d+), (\d+)\)/g;
//     const regex2 = /__FRAND\((\d+), (\d+)\)/g;
//     const regex3 = /__DATASET__/g;

//     const randomInt = (min: number, max: number) => {
//         return Math.floor(Math.random() * (max - min + 1)) + min;
//     }

//     const randomFloat = (min: number, max: number) => {
//         return Math.random() * (max - min) + min;
//     }

//     args = args.replace(regex, (match, p1, p2) => {
//         const min = parseInt(p1);
//         const max = parseInt(p2);
//         return randomInt(min, max).toString();
//     });
    
//     args = args.replace(regex2, (match, p1, p2) => {
//         const min = parseFloat(p1);
//         const max = parseFloat(p2);
//         return randomFloat(min, max).toString();
//     });

//     args = args.replace(regex3, ds_name);
//     return args;
// }


function processArgs(args: string, variables: Record<string, string>) {
    // replace __{VAR}__ with the variable value
    // replace __IRAND(MIN, MAX) with a random integer between MIN and MAX
    // replace __FRAND(MIN, MAX) with a random float between MIN and MAX

    console.log("Processing args: ", args);

    const var_regex = /__(\w+)__/g;
    const irand_regex = /__IRAND\((\d+), (\d+)\)__/g;
    const frand_regex = /__FRAND\((\d+), (\d+)\)__/g;

    const randomInt = (min: number, max: number) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    const randomFloat = (min: number, max: number) => {
        return Math.random() * (max - min) + min;
    }

    args = args.replace(var_regex, (match, p1) => {
        console.log("Replacing variable: ", p1);
        const value = variables[p1];
        if (value) {
            return value;
        } else {
            return match;
        }
    });

    args = args.replace(irand_regex, (match, p1, p2) => {
        const min = parseInt(p1);
        const max = parseInt(p2);
        return randomInt(min, max).toString();
    });


    args = args.replace(frand_regex, (match, p1, p2) => {
        const min = parseFloat(p1);
        const max = parseFloat(p2);
        return randomFloat(min, max).toString();
    });

    return args;
}

type Args = Record<string, string>;
type PipelineButtonProps = {
    variables?: Args;
    args?: Args;
}

export function PipelineButton(props: PipelineButtonProps) {
    const { data: _templates } = usePipelineTemplates();

    const templates = _templates || [
        {name: "----"}
    ];

    const [selectedTemplate, setSelectedTemplate] = React.useState<PipelineTemplate | null>(null);
    const [templateParams, setTemplateParams] = React.useState<Record<string, string>>({});
    const [
        settingsOpen,
        setSettingsOpen
    ] = React.useState(false);

    
    React.useEffect(() => {
        if (_templates && _templates.length > 0) {
            setSelectedTemplate(prev => prev ? prev : _templates[0]);
        }
    }, [_templates]);

    React.useEffect(() => {
        if (selectedTemplate) {
            setTemplateParams(selectedTemplate.parameters);
        }
    }, [selectedTemplate]);

    const runPipeline = useRunPipeline();

    const handleRun = () => {
        if(!selectedTemplate) {
            return;
        }

        if (confirm("Are you sure you want to run the pipeline?")) {
            
            const args = Object.entries(templateParams).reduce((acc, [key, value]) => {
                return {
                    ...acc,
                    [key]: processArgs(value, props.variables || {}),
                }
            }, {});
            
            runPipeline.mutate({
                temp_name: "pipeline",
                args: {
                    ...args,
                    ...props.args || {},
                }
            }, {
                onSuccess: () => {},
                onError: (err) => {
                    alert("Error starting pipeline: " + err);
                }
            })
        }
    }

    return <Stack direction="row" spacing={1}>

        <Autocomplete
            size="small"
            disableClearable
            value={selectedTemplate?.name || "----"}
            sx={{width: 300}}
            options={templates.map((template) => template.name)}
            renderInput={(params) => <TextField {...params}  />}
        />

        <ButtonGroup>
            <Button disabled={!selectedTemplate} onClick={() => {
                setSettingsOpen(true);
            }}>
                <Settings />
            </Button>
            <Button 
                variant="contained"
                startIcon={<StartOutlined />}
            disabled={!selectedTemplate} onClick={handleRun}>
                Run
            </Button>
        </ButtonGroup>

        {selectedTemplate && settingsOpen && 
            <Dialog
                open={true}
                onClose={() => {
                    setSettingsOpen(false);
                }}

                fullWidth
                maxWidth="sm"
                >

                    <Typography variant="h6" sx={{ p: 2 }}>
                        {selectedTemplate.name}
                    </Typography>   

                    {props.variables &&
                    <Typography sx={{ px: 2, pb: 1}}>
                        Variables available: {Object.keys(props.variables).join(", ")}
                    </Typography>
                    }

                    <Stack spacing={1}>
                        {Object.entries(selectedTemplate.parameters).map(([key, _]) => {
                            return <Stack key={key} direction="row" spacing={2} alignItems="center">
                                <Typography variant="body1"
                                    sx={{ minWidth: 150, textAlign: "right" }}
                                >{key}</Typography>
                                <TextField
                                    size="small"
                                    value={templateParams[key]}
                                    onChange={(e) => {
                                        setTemplateParams({
                                            ...templateParams,
                                            [key]: e.target.value
                                        });
                                    }}
                                />
                            </Stack>
                        })}
                    </Stack>


                    <Stack direction="row" justifyContent="flex-end" p={1}>

                        <Button 
                            variant="contained"
                            color="error"
                            onClick={() => {
                                setSettingsOpen(false);
                            }}
                            >
                                Exit
                            </Button>
                    </Stack>



                </Dialog>
        }


    </Stack>
}
