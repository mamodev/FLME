import { Module } from "../../backend/interfaces";

export type IEventFetchModel = {
  type: "fetch";
  client: [number, number];
  cause_aggregation?: boolean;
};

type ITrainOptimizer = {
  type: "sgd",
  learning_rate: number;
  momentum?: number;
  weight_decay?: number;
};

type ITrainParams = {
  batch_size: number;
  ephocs: number;
  optimizer: ITrainOptimizer;
};

export type IEventTrainModel = {
  type: "train";
  client: [number, number];
  cause_aggregation?: boolean;
};

export type IEventSendModel = {
  type: "send";
  client: [number, number];
  train_params: ITrainParams;
  cause_aggregation?: boolean;
};


export type IEvent = IEventFetchModel | IEventTrainModel | IEventSendModel

export type ITimeline = {
  events: IEvent[][]
  aggregations: number[]
}

export type IUpdate = {
    client: [number, number];
    tick: number;
}

export type ISimulation = {
  npartitions: number;
  client_per_partition: number[];
  proportionalKnowledge: boolean
  naggregations: number;
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