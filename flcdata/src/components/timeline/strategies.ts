import { ISimulation, ITimeline, IUpdate } from "./types";

export function computeAggregationsStats(sim: ISimulation, timeline: ITimeline): Record<number, Record<number, number>> {
    
    const clients_contribution: Record<number, Record<number, number>> = {};

    let updates: IUpdate[] = [];
    const events = timeline.events;
    
    for (let t = 0; t < events.length; t++) {
        const upds = events[t].filter((e) => e.type === "send").map((e) => ({
            client: e.client,
            tick: t
        }));

        if (upds.length > 0) {
            updates.push(...upds);
        }

        if (timeline.aggregations.includes(t)) {
            for (const u of updates) {
                if(!(u.client[0] in clients_contribution)) {
                    clients_contribution[u.client[0]] = {};
                }
                if(!(u.client[1] in clients_contribution[u.client[0]])) {
                    clients_contribution[u.client[0]][u.client[1]] = 0;
                }

                clients_contribution[u.client[0]][u.client[1]] += 1;
            }

            updates = []
        }
    }

    return clients_contribution

}
