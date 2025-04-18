import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiurl } from "./backend";



type ModuleConfig = {
    name: string;
    parameters: Record<string, any>;    
}

export type SaveRequest = {
    data_generator: ModuleConfig;
    distribution: ModuleConfig;
    partitioner: ModuleConfig;
    transformers: ModuleConfig[];
    file_name: string;
}

export function useSave(callback: () => void) {
    const queryClient = useQueryClient()
    return useMutation({
        onSuccess: (_, data) => {
            queryClient.invalidateQueries({ queryKey: ['saved-files'] })
            queryClient.invalidateQueries({ queryKey: ['data', data.file_name] })
            callback()
        },

        mutationFn: async (data: SaveRequest) => {

            const response = await fetch(apiurl("save"), {
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

export type SavedFile = {
    name: string;
    n_samples: number;
    n_classes: number;
    n_partitions: number;
    n_features: number;
    generation_params: {
        [key: string]: any
    }
}

export type SavedFilesResponse = SavedFile[]

export function useSavedFiles() {
    return useQuery({
        queryKey: ['saved-files'],
        queryFn: async () => {
            const response = await fetch(apiurl("saved-files"))
            if (!response.ok) {
                throw new Error('Network response was not ok')
            }
      
            return response.json() as Promise<SavedFilesResponse>
        }
    })
}

type DataResponse = {
    X: [number, number, number][];
    Y: number[];
    PP: number[];
    n_classes: number;
    n_samples: number;
    n_partitions: number;
}

export function useData(filename: string) {
    return useQuery({
        queryKey: ['data', filename],
        queryFn: async (ctx) => {
            const response = await fetch(apiurl(`data/${filename}`), {
                signal: ctx.signal,
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json() as Promise<DataResponse>
        },
        staleTime: 1000 * 60 * 5,
    })
}



export function useDeleteFile() {
    const queryClient = useQueryClient()
    return useMutation({
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['saved-files'] })
        },

        mutationFn: async (filename: string) => {
            const response = await fetch(apiurl(`data/${filename}`), {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json()
        }
    })
}    