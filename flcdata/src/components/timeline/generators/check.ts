

// function checkAsyncTimelineConstraints(timeline: ITimeline) {
//     // For each client, track their event sequence
//     const clientEvents: Record<string, { type: IEventType; tick: number }[]> = {};




  
//     timeline.forEach((events, tick) => {
//       for (const event of events) {
//         const hclient = hashClientId(event.client);
//         if (!clientEvents[hclient]) {
//           clientEvents[hclient] = [];
//         }
//         clientEvents[hclient].push({ type: event.type, tick });
//       }
//     });
  
//     for (const [clientIdStr, events] of Object.entries(clientEvents)) {
//       const clientId = Number(clientIdStr);
  
//       // 1. Events must be in order: fetch -> train -> send (possibly repeated)
//       let expected: IEventType = "fetch";
//       let lastTick = -1;
  
//       for (let i = 0; i < events.length; i++) {
//         const { type, tick } = events[i];
  
//         // 2. Events must be at least one tick apart (no two events for same client in same tick)
//         if (tick === lastTick) {
//           throw new Error(
//             `Client ${clientId} has multiple events in tick ${tick}`
//           );
//         }
//         lastTick = tick;
  
//         // 3. Enforce order: fetch -> train -> send
//         if (type !== expected) {
//           throw new Error(
//             `Client ${clientId} event order violation at tick ${tick}: expected ${expected}, got ${type}`
//           );
//         }
  
//         // 4. Advance expected state
//         if (expected === "fetch") expected = "train";
//         else if (expected === "train") expected = "send";
//         else if (expected === "send") expected = "fetch";
//       }
//     }
  
//     // 5. All events last one tick by design (no multi-tick events in this model)
//     // 6. Idle periods are allowed, so no check needed for that
  
//     return true; // If no error thrown, constraints are satisfied
//   } 


import { ITimeline } from "../types";
import { hashClientId } from "./utils";

export function checkEventOrder(timeline: ITimeline) {

    const fetchedSet = new Set<string>();
    const trainedSet = new Set<string>();

    for (let t = 0; t < timeline.length; t++) {
        const events = timeline[t];

        for (const event of events) {
            const eventType = event.type;
            const clientId = hashClientId(event.client);

            switch (eventType) {
                case "fetch":
                    if (trainedSet.has(clientId) || fetchedSet.has(clientId)) 
                        return "Error Fetching: Client " + clientId + " has already trained or fetched in this tick.";

                    fetchedSet.add(clientId);
                    break;
                case "train":
                    const fetched = fetchedSet.has(clientId);
                    const trained = trainedSet.has(clientId);
                    if (!fetched || trained) 
                        return "Error Training: Client " + clientId +  `has ${fetched ? "already trained" : "not fetched"} in this tick.`;
                    
                    trainedSet.add(clientId);
                    fetchedSet.delete(clientId);
                    break;
                case "send":
                    if (!trainedSet.has(clientId)) 
                        return "Error Sending: Client " + clientId + " has not trained in this tick.";
                    trainedSet.delete(clientId);
                    break;
                default:
                    throw new Error(`Unknown event type: ${eventType}`);
            }
        }
    }

    return null;
}







