import {
  ISimulation,
  ITimeline,
  ClientState,
  IEvent,
  IUpdate,
  IModule,
} from "../types";
import { getAllClientIds, hashClientId, shuffleArray } from "./utils";

type Param = {
  fetch_prob: number;
  train_prob: number;
  send_prob: number;
  min_idle_after_train: number;
  max_idle_after_train: number;
  nbuff: number; // Buffer size, not used in this generator
};

function buffStrat(n: number) {

  if (typeof n !== "number" || n <= 0) {
    throw new Error("Buffer size must be a positive integer");
  }

  return function (updates: IUpdate[]): boolean {
    console.log(`Buffer strategy: checking if updates length ${updates.length} >= ${n}`);
    return updates.length >= n;
  };
}

export function generate(simulation: ISimulation, params: Param): ITimeline {
  const {
    fetch_prob,
    train_prob,
    send_prob,
    min_idle_after_train,
    max_idle_after_train,
  } = params;

  const strategy = buffStrat(params.nbuff);

  const { client_per_partition } = simulation;
  let allClients = getAllClientIds(client_per_partition);

  const timeline: ITimeline = {
    events: [],
    aggregations: [],
  }

  const clientStates: Record<string, ClientState> = {};
  const idleAfterTrain: Record<string, number> = {};

  let updates: IUpdate[] = [];

  while (timeline.aggregations.length < simulation.naggregations) {
    const events: IEvent[] = [];

    allClients = shuffleArray(allClients);
    for (const clientId of allClients) {
      const chash = hashClientId(clientId);
      const state = clientStates[chash] ?? "idle";

      switch (state) {
        case "idle":
          if (Math.random() < fetch_prob) {
            events.push({ type: "fetch", client: clientId });
            clientStates[chash] = "fetched";
          }
          break;
        case "fetched":
          if (Math.random() < train_prob) {
            events.push({ type: "train", client: clientId });
            clientStates[chash] = "trained";

            idleAfterTrain[chash] =
              min_idle_after_train +
              Math.floor(
                Math.random() *
                  (max_idle_after_train - min_idle_after_train + 1)
              );
          }
          break;
        case "trained":
          if (!(chash in idleAfterTrain)) {
            throw new Error(`Client ${clientId} not found in idleAfterTrain`);
          }

          if (idleAfterTrain[chash] > 0) {
            idleAfterTrain[chash]--;
            break;
          }

          if (Math.random() < send_prob) {
            events.push({ type: "send", client: clientId });
            delete clientStates[chash];
            delete idleAfterTrain[chash];

            updates.push({
              client: clientId,
              tick: timeline.events.length,
            });

            if (strategy(updates)) {
              updates = [];
              events[events.length - 1].cause_aggregation = true;
            }
          }
          break;

        default:
          throw new Error(`Unknown state: ${state}`);
      }

     
    }

    const sendEvents = events.filter((event) => event.type === "send");
    const eventsWithoutSend = events.filter(
      (event) => event.type !== "send"
    ) as IEvent[];

    if (sendEvents.length > 0) {
      eventsWithoutSend.push(sendEvents[0]);
      if (sendEvents[0].cause_aggregation) {
        timeline.aggregations.push(timeline.events.length);
      } 
    }

    timeline.events.push(eventsWithoutSend);
    

    if (sendEvents.length > 1) {
      for (let i = 1; i < sendEvents.length; i++) {
        timeline.events.push([sendEvents[i]]);
        if (sendEvents[i].cause_aggregation) {
          timeline.aggregations.push(timeline.events.length - 1);
        }
      }
    }
  }

  return timeline;
}

export const TLRandomAsync: IModule = {
  fn: generate,
  name: "Random Async",
  description: "Randomly select clients to fetch, train, and send updates.",
  parameters: {
    nbuff: {
      type: "int",
      default: 5,
      min: 1,
      max: 1000,
    },
    fetch_prob: {
      type: "float",
      min: 0,
      max: 1,
      default: 0.1,
    },
    train_prob: {
      type: "float",
      min: 0,
      max: 1,
      default: 0.5,
    },
    send_prob: {
      type: "float",
      min: 0,
      max: 1,
      default: 0.5,
    },
    min_idle_after_train: {
      type: "int",
      min: 0,
      max: 10,
      default: 1,
    },
    max_idle_after_train: {
      type: "int",
      min: 0,
      max: 10,
      default: 3,
    },

  },
};
