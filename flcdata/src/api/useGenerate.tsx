import { useMutation, useQuery } from "@tanstack/react-query";
import { apiurl } from "./backend";



type ModuleConfig = {
    name: string;
    parameters: Record<string, any>;    
}

export type GenerateRequest = {
    data_generator: ModuleConfig;
    distribution: ModuleConfig;
    partitioner: ModuleConfig;
}

export function useGenerate(callback: (data: any) => void) {
    return useMutation({
        onSuccess: (data) => {
            callback(data)
        },

        mutationFn: async (data: GenerateRequest) => {

            const response = await fetch(apiurl("generate"), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json()
        }
    })
}