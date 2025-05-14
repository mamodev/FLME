import { ISimulation, ITimeline, IUpdate } from "./types";


export function computeAggregations(sim: ISimulation, timeline: ITimeline) {

      const aggticks: number[] = []
    
      let updates: IUpdate[] = [];
      for (let t = 0; t < timeline.length; t++) {
          const upds = timeline[t].filter((e) => e.type === "send").map((e) => ({
              client: e.client,
              tick: t
          }));
  
          if (upds.length > 0) {
              updates.push(...upds);
          }
  
          if (sim.strategy(updates)) {
        
              aggticks.push(t)
              updates = []
          }
      }

      return aggticks
}


export function computeAggregationsStats(sim: ISimulation, timeline: ITimeline): Record<number, Record<number, number>> {
    
    const clients_contribution: Record<number, Record<number, number>> = {};

    let updates: IUpdate[] = [];
    for (let t = 0; t < timeline.length; t++) {
        const upds = timeline[t].filter((e) => e.type === "send").map((e) => ({
            client: e.client,
            tick: t
        }));

        if (upds.length > 0) {
            updates.push(...upds);
        }

        if (sim.strategy(updates)) {
    
            for (const u of updates) {
                if(!(u.client[0] in clients_contribution)) {
                    clients_contribution[u.client[0]] = {};
                }
                if(!(u.client[1] in clients_contribution[u.client[0]])) {
                    clients_contribution[u.client[0]][u.client[1]] = 0;
                }

                clients_contribution[u.client[0]][u.client[1]] += 1;
            }

            // 
            updates = []
        }
    }

    return clients_contribution

}
