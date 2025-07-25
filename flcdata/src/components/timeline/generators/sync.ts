import {
  ISimulation,
  ITimeline,
  IModule,
  IEvent,
} from "../types";
import { getAllClientIds, shuffleArray } from "./utils";

type Param = {
  drp_prob_min: number;
  drp_prob_max: number;
  client_per_round: number;

  allow_partial_aggregation: boolean;

  // Training parameters
  ephocs: number;
  batch_size: number;
  learning_rate: number;
  momentum: number;
  weight_decay: number;
}

function generate(simulation: ISimulation, params: Param): ITimeline {

  const { client_per_partition } = simulation;
  let allClients = getAllClientIds(client_per_partition);

  const timeline: ITimeline = {
    events: [],
    aggregations: [],
  }

  if (params.drp_prob_min < 0 || params.drp_prob_min > 1 ||
      params.drp_prob_max < 0 || params.drp_prob_max > 1 ||
      params.drp_prob_min > params.drp_prob_max) {
        alert("Invalid dropout probabilities");
        return timeline;
  }

  while (timeline.aggregations.length < simulation.naggregations) {

    allClients = shuffleArray(allClients);
    const selectedClients = allClients.slice(0, params.client_per_round);

    const drp_prob = Math.random() * (params.drp_prob_max - params.drp_prob_min) + params.drp_prob_min;


    const nActiveClients = Math.ceil(selectedClients.length * (1 - drp_prob));
    const activeClients = selectedClients.slice(0, nActiveClients);
    const inactiveClients = selectedClients.slice(nActiveClients, selectedClients.length);

    let evts: IEvent[] = [];
    
    for (const clientId of (params.allow_partial_aggregation ? selectedClients : activeClients)) {
      evts.push({ type: "fetch", client: clientId });
    }

    timeline.events.push(evts);

    evts = [];
    
    for (const clientId of (params.allow_partial_aggregation ? selectedClients : activeClients)) {
      evts.push({ type: "train", client: clientId});
    }

    timeline.events.push(evts);
    

    evts = [];
    for (const clientId of activeClients) {
      evts.push({ type: "send", client: clientId, 
        train_params: {
          batch_size: params.batch_size,
          ephocs: params.ephocs,
          optimizer: {
            type: "sgd",
            learning_rate: params.learning_rate,
            momentum: params.momentum,
            weight_decay: params.weight_decay,
          }
        },
      });
    } 

    timeline.events.push(evts);

    if (params.allow_partial_aggregation && inactiveClients.length !== 0) {

      evts = [];
      
      for (const clientId of inactiveClients) {
        evts.push({
          type: "send",
          client: clientId,
          train_params: {
            batch_size: params.batch_size,
            // ephocs: rand between 1 and params.ephocs
            ephocs: Math.floor(Math.random() * params.ephocs) + 1,
            optimizer: {
              type: "sgd",
              learning_rate: params.learning_rate,
              momentum: params.momentum,
              weight_decay: params.weight_decay,  
            }
          },
        });
      }

      timeline.events.push(evts);


    }





    timeline.aggregations.push(timeline.events.length - 1);
  }


  return timeline;
}


export const TLSync: IModule = {
  fn: generate,
  name: "sync",
  description: "Synchronous timeline generator",
  parameters: {

    client_per_round: {
      type: "float",
      default: 10,
      min: 0,
      max: 1,
    },

    allow_partial_aggregation: {
      type: "boolean",
      default: true,
    },

    drp_prob_min: {
      type: "float",
      default: 0.1,
      min: 0,
      max: 1,
    },

    drp_prob_max: {
      type: "float",
      default: 0.2,
      min: 0,
      max: 1,
    },

    ephocs: {
      type: "int",
      default: 20,
      min: 1,
      max: 100,
    },

    batch_size: {
      type: "int",
      default: 10,
      min: 1,
      max: 1000,
    },

    learning_rate: {
      type: "float",
      default: 0.01,
      min: 0.0001,
      max: 1,
    },

    momentum: {
      type: "float",
      default: 0,
      min: 0,
      max: 1,
    },

    weight_decay: {
      type: "float",
      default: 0,
      min: 0,
      max: 1,
    },
  }
};