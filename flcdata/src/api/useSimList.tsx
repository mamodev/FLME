import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiurl } from "./backend";


type SimulationInfo = {
    name: string;
    model_files: number
    metrics_files: number
    info: {
        "dataset": string
    }
}

type SimulationsInfo = SimulationInfo[];    

export function useSimList() {
    return useQuery({
        queryKey: ['sim-list'],
        queryFn: async (ctx) => {
            const response = await fetch(apiurl('list-simulations'), {
                signal: ctx.signal,  
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json() as Promise<SimulationsInfo>
        },
        staleTime: 1000 * 60 * 5,
    })
}


export function useDeleteSimulation() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async (simulationName: string) => {
            const response = await fetch(apiurl(`simulation/${simulationName}`), {
                method: 'DELETE',
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json()
        },
        onSuccess: () => {
            // Invalidate the query to refetch the simulation list
            queryClient.invalidateQueries({ queryKey: ['sim-list'] })
        }
    })
}