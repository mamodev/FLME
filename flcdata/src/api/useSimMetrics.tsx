import { useQuery } from "@tanstack/react-query";
import { apiurl } from "./backend";

type SimulationMetric = {
    version: number;
    accuracy: number;
    groups: {
        [key: string]: number;
    };

    contributors: {
        auth: {
            cluster: number;
            exp: number;
            gid: number;
            key: string;
            pid: number;
        };
        meta: {
            learning_rate: number;
            local_epoch: number;
            momentum: number;
            test_loss: number;
            train_loss: number;
            train_samples: number;
        };
    }[];
}

type SimulationMetrics = SimulationMetric[];    

export function useSimMetrics(sim_name: string) {
    return useQuery({
        queryKey: ['sim-metrics', sim_name],
        queryFn: async (ctx) => {
            const response = await fetch(apiurl(`simulation-metrics/${sim_name}`), {
                signal: ctx.signal,  
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json() as Promise<SimulationMetrics>
        },
        staleTime: 1000 * 60 * 5,
    })
}