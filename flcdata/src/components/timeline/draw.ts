import { grey, blue, green, orange, red } from "@mui/material/colors";
import { hashClientId } from "./generators/utils";
import { ISimulation, ITimeline, IUpdate } from "./types";

const palette = [
  'red',
  'blue',
  'green',
  'orange',
  'purple',
  'pink',
  'brown',
  'grey',
  'cyan',
  'lime',
  'indigo',
  'teal',
  'violet',
  'magenta',
  'gold',
  'salmon',
];

// const palette = ["#ff0000", "#fc2a00", "#f93d00", "#f64b00", "#f35700", "#f06200", "#ed6b00", "#e97400", "#e57d00", "#e28500", "#de8c00", "#d99300", "#d59a00", "#d0a100", "#cba800", "#c6ae00", "#c0b500", "#babb00", "#b4c100", "#adc700", "#a6cd00", "#9ed200", "#96d800", "#8cde00", "#81e400", "#75e900", "#67ef00", "#55f400", "#3dfa01", "#00ff04"]


function padding(
  x: number,
  y: number,
  w: number,
  h: number,
  padding: number | [number, number]
): [number, number, number, number] {
  const px = typeof padding === "number" ? padding : padding[0];
  const py = typeof padding === "number" ? padding : padding[1];

  return [x + px, y + py, w - 2 * px, h - 2 * py] as const;
}

function flex(
  x: number,
  y: number,
  w: number,
  h: number,
  flex: number[]
): [number, number, number, number][] {
  const total = flex.reduce((a, b) => a + b, 0);
  flex = flex.map((f) => f / total);

  const result: [number, number, number, number][] = [];
  let offsetX = x;
  let offsetY = y;
  let offsetW = w;
  for (let i = 0; i < flex.length; i++) {
    const f = flex[i];
    const w = offsetW * f;
    result.push([offsetX, offsetY, w, h]);
    offsetX += w;
  }

  return result;
}

const colors = {
  fetch: grey[200],
  train: grey[400],
  send: grey[800],
};

type Activity = {
  client: [number, number];
  start: number;
  end: number;
  startTrain: number;
  offsY: number;
};

type DrawPreComputed = {
  activities: Activity[]
  maxActivities: number 
  aggregations: number[]
}


export function precompute(timeline: ITimeline, sim: ISimulation): DrawPreComputed {
  const nclients = sim.client_per_partition.reduce(
    (a, b) => a + b,
    0
  );

  const activities: Activity[] = [];
  const clientStates: Record<string, Activity> = {};
  const busyOffsets = new Set<number>();

  let maxActivities = 0;

  function getFirstFreeOffset() {
    for (let i = 0; i < nclients; i++) {
      if (!busyOffsets.has(i)) {
        if (i > maxActivities) {
          maxActivities = i;
        }

        busyOffsets.add(i);
        return i;
      }
    }

    throw new Error("No free offset available");
  }


  
  for (let t = 0; t < timeline.events.length; t++) {
    const evts = timeline.events[t];

    evts.sort((a, b) => {
      if (a.type === "fetch") {
        return b.type === "fetch" ? 0 : -1;
      }

      return 1
    });

    for (let e = 0; e < evts.length; e++) {
      const evt = evts[e];
      const c = hashClientId(evt.client);

      if (!clientStates[c]) {
        if (evt.type === "fetch") {
          clientStates[c] = {
            client: evt.client,
            start: t,
            end: t,
            startTrain: -1,
            offsY: getFirstFreeOffset(),
          };
        }

        continue;
      }

      if (evt.type === "train") {
        clientStates[c].startTrain = t;
      }

      if (evt.type === "send") {
        const a = clientStates[c];
        a.end = t;
        busyOffsets.delete(a.offsY);
        activities.push(clientStates[c]);
        delete clientStates[c];
      }
    }
  } 

  const aggregations: number[] = timeline.aggregations

  return {
    activities,
    maxActivities, 
    aggregations
  }

}

export function draw(canvas: HTMLCanvasElement, timeline: ITimeline, sim: ISimulation,
  precomputed: DrawPreComputed,
  time_start=0,
  time_end=timeline.events.length - 1,
) {
    const {maxActivities, activities, aggregations} = precomputed
  
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to get canvas context");
    }
  

    const TOP_OFFS = 30

    ctx.globalAlpha = 1
    ctx.fillStyle = grey[100];
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = grey[500];
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);


    ctx.fillStyle = grey[400];
    ctx.fillRect(0, 0, canvas.width, TOP_OFFS - 10);

    // time_start : timelen = x : canvaswidth
    // time_span : timelen = x : canvaswidth

    ctx.fillStyle = blue[500];
    ctx.fillRect(time_start * canvas.width / timeline.events.length, 0, 
        (time_end - time_start) *canvas.width / timeline.events.length
      , TOP_OFFS - 10);


  
  

    const canvas_height = canvas.height - TOP_OFFS
  
    const w_tick = canvas.width / (time_end - time_start + 1);
    const h_client = Math.min(canvas_height / maxActivities, 20);
  
   
  
    // vertical dash lines for each tick
    // ctx.setLineDash([2, 2]);
    // ctx.strokeStyle = grey[300];
    // ctx.lineWidth = 1;

    // for (let i = 0; i < timeline.length; i++) {
    //   const x = i * w_tick;
    //   ctx.beginPath();
    //   ctx.moveTo(x, TOP_OFFS - 10);
    //   ctx.lineTo(x, canvas_height + TOP_OFFS )
    //   ctx.stroke();
    // }
    // ctx.setLineDash([]);

    
   
    for (const { start, end, startTrain, client, offsY } of activities) {
      // if (start < time_start || end > time_end) {
      //   continue;
      // }
      if (!(end > time_start && start < time_end)) {
        continue
      }


      const y = offsY * h_client + TOP_OFFS
      const x = (start - time_start) * w_tick;
      const w = (end - start + 1)  * w_tick;
      const h = h_client;
  
      // const box = padding(x, y, w, h, [5, 2]);
      const box = padding(x, y, w, h, 2);

        
      ctx.globalAlpha = 1
      ctx.fillStyle = palette[client[0] % palette.length];
      ctx.strokeStyle = "black";
      ctx.lineWidth = 1;
  
      ctx.fillRect(...box);
      ctx.strokeRect(...box);
  
      const innerBox = box;
  
      if (time_end - time_start < 1000) {


        const flexBoxes = flex(...innerBox, [
          startTrain - start,
          end - startTrain,
          1,
        ]);
        
        ctx.globalAlpha = Math.min((1000 - (time_end - time_start)) / 1000, 0.5)
        const flexColors = [colors["fetch"], colors["train"], colors["send"]];
    
        for (let i = 0; i < flexBoxes.length; i++) {
          const [x, y, w, h] = flexBoxes[i];
          ctx.fillStyle = flexColors[i];
          ctx.strokeStyle = "black";
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x, y, w, h);
        }
        ctx.globalAlpha = 1;
      }

   
      for (const t of aggregations) {
        if(t < start) {
          continue
        }

        if(t > end) {
          break
        }

   
        const x = (t - time_start) * w_tick + w_tick / 2;
        ctx.strokeStyle = red[500];
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas_height + TOP_OFFS);
        ctx.stroke();
      }

      // let updates: IUpdate[] = [];
      // for (let t = 0; t < timeline.length; t++) {
      //     const upds = timeline[t].filter((e) => e.type === "send").map((e) => ({
      //         client: e.client,
      //         tick: t
      //     }));
  
      //     if (upds.length > 0) {
      //         updates.push(...upds);
      //     }
  
      //     if (sim.strategy(updates)) {
      //         const x = t * w_tick;
      //         ctx.strokeStyle = red[500];
      //         ctx.lineWidth = 1;
      //         ctx.beginPath();
      //         ctx.moveTo(x, 0);
      //         ctx.lineTo(x, canvas_height);
      //         ctx.stroke();
          
      //         updates = []
      //     }
      // }
    }
  }