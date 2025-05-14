import { IEvent, ISimulation, ITimeline } from "../types";
import { pickNRandomIntegers } from "./utils";

export function generateFlSyncTimeline(
  simulation: ISimulation,
  adhesion_ratio = 0.1
): ITimeline {
  const timeline: ITimeline = [];
  const { npartitions, client_per_partition, naggregations } = simulation;

  // randomly pick a client from each partition
  const clients = Array.from(
    { length: npartitions },
    (_, i) => client_per_partition[i]
  );
  const nclients = clients.reduce((a, b) => a + b, 0);

  const nclients_per_partition = Math.floor(
    (nclients * adhesion_ratio) / npartitions
  );
  // const nclients_per_aggregation = nclients_per_partition * npartitions;

  for (let i = 0; i < naggregations; i++) {
    const partitionClients: number[][] = [];

    for (let j = 0; j < npartitions; j++) {
      partitionClients.push(
        pickNRandomIntegers(0, clients[j] - 1, nclients_per_partition)
      );
    }

    let evnts: IEvent[] = [];
    for (let j = 0; j < npartitions; j++) {
      for (let k = 0; k < partitionClients[j].length; k++) {
        evnts.push({ type: "fetch", client: [j, k] });
      }
    }

    timeline.push(evnts);
    evnts = [];

    for (let j = 0; j < npartitions; j++) {
      for (let k = 0; k < partitionClients[j].length; k++) {
        evnts.push({ type: "train",  client: [j, k] });
      }
    }

    timeline.push(evnts);
    evnts = [];
    for (let j = 0; j < npartitions; j++) {
      for (let k = 0; k < partitionClients[j].length; k++) {
        evnts.push({ type: "send",  client: [j, k] });
      }
    }
    timeline.push(evnts);
  }
  return timeline;
}
