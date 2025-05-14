import { useQuery } from "@tanstack/react-query";
import { apiurl } from "./backend";

import type { Config } from "../backend/interfaces";

export function useConfig() {
    const url = apiurl('config')

    return useQuery({
        queryKey: ['config'],
        queryFn: async (ctx) => {
            console.log('Fetching config from', url)
            const response = await fetch(url, {
                signal: ctx.signal,  
            })

            if (!response.ok) {
                throw new Error('Network response was not ok')
            }

            return response.json() as Promise<Config>
        },
        staleTime: 1000 * 60 * 5,
    })
}