
import { Box } from "@mui/material"
import Plotly from 'plotly.js'
import React from "react"

type PlotRendererProps = {
  data: any
  onPointClick?: (point: number, trace: number) => void
}

export function PlotRenderer(props: PlotRendererProps) {

    const { data } = props
  
    const ref = React.useRef<HTMLDivElement>(null)
  
    React.useEffect(() => {
      if (!ref.current) return
      const div = ref.current
  
      const layout = {}
      const config = {
        responsive: true,
      }
  
      Plotly.newPlot(div, data, layout, config).then(() => {
        if(!props.onPointClick) return
        div.on('plotly_click', (event: any) => {
          if(!props.onPointClick) return
          const point = event.points[0]
          props.onPointClick(point.pointNumber, point.curveNumber)
        })
      })

  
      return () => {
        Plotly.purge(div)
      }
  
    }, [ref.current, data])
  
  
    return <Box ref={ref} sx={{ height: "100%", flex: 1, boxSizing: "border-box",}}/>
  }