import React from "react";
import Box from "@mui/material/Box";
import { PlotlyConfig, PlotlyCustomDowload, PlotlyLayout, usePlotly } from "../contexts/PlotlyCtx";

type PlotRendererProps = {
  data: Plotly.Data[];
  onPointClick?: (point: number, trace: number) => void;
  chunkSize?: number; // how many points per trace
  layout?: Partial<PlotlyLayout>;
};

function splitTrace(trace: Plotly.Data, chunkSize: number): Plotly.Data[] {
  const { x, y, z, ...rest } = trace as any;
  const length = x.length;
  const traces: Plotly.Data[] = [];
  for (let i = 0; i < length; i += chunkSize) {
    traces.push({
      x: x.slice(i, i + chunkSize),
      y: y.slice(i, i + chunkSize),
      z: z ? z.slice(i, i + chunkSize) : undefined,
      ...rest,
    });
  }
  return traces;
}

function interleaveChunks(
  traces: Plotly.Data[],
  chunkSize: number
): Plotly.Data[] {
  const splitTraces = traces.map((trace) => splitTrace(trace, chunkSize));
  const maxChunks = Math.max(...splitTraces.map((chunks) => chunks.length));
  const interleaved: Plotly.Data[] = [];

  for (let i = 0; i < maxChunks; i++) {
    for (let t = 0; t < splitTraces.length; t++) {
      if (splitTraces[t][i]) {
        interleaved.push(splitTraces[t][i]);
      }
    }
  }
  return interleaved;
}

export function PlotRenderer(props: PlotRendererProps) {
  const { data, onPointClick, chunkSize = 500 } = props;
  const ref = React.useRef<HTMLDivElement>(null);

  const { Plotly } = usePlotly();
  
  React.useEffect(() => {
    if (!ref.current) return;
    const div = ref.current as HTMLDivElement & {
      on: (event: string, callback: (event: any) => void) => void;
    };

    const layout = {
      ...(props.layout || {}),
    }

    const config: Partial<PlotlyConfig> = { responsive: true,
      modeBarButtonsToAdd: [
        PlotlyCustomDowload,
      ],

    };

    let unmounted = false;

    // Only split traces if 3D (has z)
    let traces: Plotly.Data[] = [];
    if (data.length > 0 && "z" in data[0]) {
      traces = interleaveChunks(data, chunkSize);
    } else {
      traces = data;
    }


    Plotly.newPlot(div, [], layout, config).then(() => {
      if (onPointClick) {
        div.on("plotly_click", (event: any) => {
          const point = event.points[0];
          onPointClick(point.pointNumber, point.curveNumber);
        });
      }

      const addTracesAsync = async () => {
        for (let i = 0; i < traces.length; i += data.length) {
          if (unmounted) break;
          await new Promise((resolve) => {
            if ('requestIdleCallback' in window) {
              (window as any).requestIdleCallback(resolve);
            } else {
              requestAnimationFrame(resolve);
            }
          });
          await Plotly.addTraces(div, traces.slice(i, i + data.length));
        }
      };
      addTracesAsync();
    });

    return () => {
      unmounted = true;
      Plotly.purge(div);
    };
  }, [data, onPointClick, chunkSize]);

  return (
    <Box ref={ref} sx={{ height: "100%", flex: 1, boxSizing: "border-box" }} />
  );
}
