import { useQuery } from "@tanstack/react-query";
import { apiurl } from "./backend";

import type { Config } from "../backend/interfaces";

export function useConfig() {
    return useQuery({
        queryKey: ['config'],
        queryFn: async (ctx) => {
            const response = await fetch(apiurl('config'), {
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