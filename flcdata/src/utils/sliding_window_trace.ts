import { PlotlyData } from "../contexts/PlotlyCtx";

export interface SlidingTracesOpts {
  x: number[];
  y: number[] | number[][];
  windowSize: number;
  bandFillColor?: string;
  label?: string;
  avg_color?: string;
  raw_color?: string;
  var_color?: string;
  min_max_color?: string; // Color for the min-max band
}



type SlidingWindowStats = {
  xmids: number[];
  means: number[];
  stds: number[];
  mins: number[];
  maxs: number[];
};

function slidingStats(
  x: number[],
  y: number[][] | number[],
  w: number
): SlidingWindowStats{
  const N = y.length;
  if (w > N) {
    throw new Error("Window size must be ≤ data length");
  }

  const xmids: number[] = [];
  const means: number[] = [];
  const stds: number[] = [];
  const mins: number[] = [];
  const maxs: number[] = [];


  let cursor = 0;
  while (cursor + w <= N) {
    const _slice = y.slice(cursor, cursor + w);
    xmids.push(
      x[cursor + Math.floor(w / 2)]
    );

    const flat_slice = _slice.flat();

    const mean = flat_slice.reduce((a, b) => a + b, 0) / flat_slice.length;
    means.push(mean);

    const variance =
      flat_slice.reduce((acc, v) => acc + (v - mean) ** 2, 0) / flat_slice.length;
    const std = Math.sqrt(Math.max(0, variance));
    stds.push(std);

    const min = Math.min(...flat_slice);
    mins.push(min);

    const max = Math.max(...flat_slice);
    maxs.push(max);

    cursor = cursor + Math.floor(w / 2);
  }

  return {
    xmids,
    means,
    stds,
    mins,
    maxs, 
  };
}


export function createSlidingWindowTraces(
  opts: SlidingTracesOpts
): Plotly.Data[] {
  const { x, y, windowSize: w } = opts;
  const {
    xmids,
    means,
    stds,
    mins,
    maxs,
  } = slidingStats(x, y, w);

  // Build the ±1σ band polygon
  const upper = means.map((m, i) => m + stds[i]);
  const lower = means.map((m, i) => m - stds[i]).reverse();
  const xBand = xmids.concat(xmids.slice().reverse());
  const yBand = upper.concat(lower);

  const upperMinMax = maxs;
  const lowerMinMax = mins.slice().reverse();
  const xBandMinMax = xmids.concat(xmids.slice().reverse());
  const yBandMinMax = upperMinMax.concat(lowerMinMax);

  const var_color = opts.var_color || "rgba(200,50,50,0.5)";
  const raw_color = opts.raw_color || "rgba(0,100,200,0.5)";
  const avg_color = opts.avg_color || "rgba(0,100,200,1)";
  const min_max_color = opts.min_max_color || "rgba(50,50,50,0.1)"; 

  const label = opts.label || "";

  const traceBand: Plotly.Data = {
    x: xBand,
    y: yBand,
    fill: "toself",
    fillcolor: var_color,
    line: { color: "transparent" },
    name: `${label} ±1σ`,
    hoverinfo: "skip",
    type: "scatter"
  };

  const minMaxBand: Plotly.Data = {
    x: xBandMinMax,
    y: yBandMinMax,
    fill: "toself",
    fillcolor: min_max_color,
    line: { color: "transparent" },
    name: `${label} min - max`,
    hoverinfo: "skip",
    type: "scatter",
  };




  const traceMean: Plotly.Data = {
    x: xmids,
    y: means,
    mode: "lines",
    line: {
        color: avg_color,
        width: 3
    },
    // name: "Running mean",
    name: `${label} running mean (${w})`,
    type: "scatter"
  };

  const originalTrace: Plotly.Data = {
    x,
    y,
    mode: "lines",
    name: `${label} raw data`,
    visible: "legendonly",
    line: {
        color: raw_color,
        width: 1
    },
   };

    return [minMaxBand, originalTrace, traceBand, traceMean,  ];
}


export interface TracesOptions {
  showBand?: boolean;
  bandColor?: string;
  meanColor?: string;
}

/**
 * Generate Plotly traces for mean ± standard deviation.
 *
 * @param x - array of x values
 * @param y - array of y-arrays, one per x
 * @param opts - optional settings
 * @returns array of Plotly.Data traces
 */
export function makeMeanStdevTraces(
  x: number[],
  y: number[][],
  opts: TracesOptions = {}
): PlotlyData[] {
  const {
    showBand = true,
    bandColor = "rgba(0,0,255,0.2)",
    meanColor = "blue",
  } = opts;


  if (x.length !== y.length) {
    throw new Error("x and y must have the same length");
  }



  // compute means & standard deviations
  const means = y.map(arr => {
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum / arr.length;
  });
  const stds = y.map((arr, i) => {
    const μ = means[i];
    const variance =
      arr.reduce((acc, v) => acc + (v - μ) ** 2, 0) / arr.length;
    return Math.sqrt(Math.max(0, variance));
  });

  // trace: mean line + error bars
  const traceMean: PlotlyData = {
    x,
    y: means,
    mode: "lines+markers",
    name: "Mean ±1σ",
    line: { color: meanColor },
    error_y: {
      type: "data",
      array: stds,
      visible: true,
      thickness: 1.5,
      width: 4,
      color: bandColor.replace(/0\.[0-9]+\)$/, "0.3)"),
    },
  };

  if (!showBand) {
    return [traceMean];
  }

  // optional shaded σ‐band
  const upper: PlotlyData = {
    x,
    y: means.map((m, i) => m + stds[i]),
    mode: "lines",
    line: { width: 0 },
    showlegend: false,
    hoverinfo: "skip",
  };
  const lower: PlotlyData = {
    x,
    y: means.map((m, i) => m - stds[i]),
    mode: "lines",
    fill: "tonexty",
    fillcolor: bandColor,
    line: { width: 0 },
    showlegend: false,
    hoverinfo: "skip",
  };

  return [upper, lower, traceMean];
}
