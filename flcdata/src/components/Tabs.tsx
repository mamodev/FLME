
import { Stack } from "@mui/material"
import React from "react"


export function TabContent(props: {children: React.ReactNode | React.ReactNode[]
    current: string | number, value: string | number}) {
    
    const { children, current, value } = props
  
    if (current !== value) {
      return <></>
    }
  
    return <Stack direction={"row"} flex={1}
      sx={{
        position: "relative",
        boxSizing: "border-box",
        overflow: 'auto'
      }}
  
    >
      {children}
    </Stack>
  }
  