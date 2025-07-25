import { Stack, TextField } from "@mui/material";

export function DatasetHubPanel() {




  return (

    <Stack flex={1}>
        <Stack direction={"row"} flex={1} justifyContent="center" >
            <TextField 
            placeholder="Search datasets"
            fullWidth sx={{maxWidth: 600}} />
        </Stack>




    </Stack>

);
}
