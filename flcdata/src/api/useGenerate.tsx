import { useMutation } from "@tanstack/react-query";
import { apiurl } from "./backend";

type ModuleConfig = {
    name: string;
    parameters: Record<string, any>;    
}

export type GenerateRequest = {
    data_generator: ModuleConfig;
    distribution: ModuleConfig;
    partitioner: ModuleConfig;
    transformers: ModuleConfig[];
}

export type GenerateResponse = {
    'X': number[][];
    'Y': number[],
    'PP': number[],
    'CTP': number[][];
    'n_classes': number,
    'n_samples': number,
    "n_partitions": number,
}

export function useGenerate(callback: (data: GenerateResponse) => void) {
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

            return response.json() as unknown as GenerateResponse
        }
    })
}