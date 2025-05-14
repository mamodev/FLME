import { Module } from "../../backend/interfaces";

export type IEventFetchModel = {
  type: "fetch";
  client: [number, number];
};

export type IEventTrainModel = {
  type: "train";
  client: [number, number];
};

export type IEventSendModel = {
  type: "send";
  client: [number, number];
};


export type IEvent = IEventFetchModel | IEventTrainModel | IEventSendModel

export type ITimeline = IEvent[][];

export type IUpdate = {
    client: [number, number];
    tick: number;
}

export type ISimulation = {
  npartitions: number;
  client_per_partition: number[];
  proportionalKnowledge: boolean
  naggregations: number;
  strategy: (updates: IUpdate[]) => boolean;
};


export type ClientState = "idle" | "fetched" | "trained" | "sent";

export type IEventType = "fetch" | "train" | "send";


export type SIM_EXPORT = { 
  simulation: ISimulation
  aggregations: number[]
}

export type IModule = Module & {
  fn: (simulation: ISimulation, params: any) => ITimeline;
}