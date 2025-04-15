import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiurl } from "./backend";


export type Pipeline = {
  id: string;
  last_status_change: number;
  pid: string;
  start_time: number;
  status: string;
  dex: string;
  config: any;
};


// Hook for GET /api/pipelines
export function usePipelines() {
  return useQuery({
    queryKey: ["pipelines"],
    queryFn: async (ctx) => {
      const response = await fetch(apiurl("pipelines"), {
        signal: ctx.signal,
      });
      if (!response.ok) {
        throw new Error("Failed to fetch pipelines");
      }
      return response.json() as Promise<Pipeline[]>;
    },
    staleTime: 1000 * 60 * 1, // 1 minute
  });
}

// Hook for DELETE /api/pipelines/:id
export function useDeletePipeline() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      const response = await fetch(apiurl("pipelines", id), {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error("Failed to delete pipeline");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
    },
  });
}


// Hook for POST /api/pipeline/:temp_name/run
export function useRunPipeline() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ temp_name, args }: { temp_name: string; args: Record<string, string> }) => {
      const response = await fetch(apiurl("pipeline", temp_name, "run"), {
        method: "POST",
        body: JSON.stringify({ args }),
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!response.ok) {
        throw new Error("Failed to run pipeline");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
    },
  });
}



// Hook for POST /api/pipelines/:pipeline_id/rerun
export function useRerunPipeline() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (pipeline_id: string) => {
      const response = await fetch(apiurl("pipeline", pipeline_id, "rerun"), {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error("Failed to rerun pipeline");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
    },
  });
}

type PipelineLog = {
  stdout: string[]
  stderr: string[]
}

// Hook for GET /api/pipelines/:pipeline_id/log
export function usePipelineLog(pipeline_id: string) {
  return useQuery({
    queryKey: ["pipelines", pipeline_id, "log"],
    queryFn: async (ctx) => {
      const response = await fetch(apiurl("pipeline", pipeline_id, "log"), {
        signal: ctx.signal,
      });
      if (!response.ok) {
        throw new Error("Failed to fetch pipeline log");
      }
      return response.json() as Promise<PipelineLog>;
    },
    staleTime: 1000 * 60 * 1, // 1 minute
  });
}


type PipelineStep = {
  type: string;
  steps: (string | PipelineStep)[];
}

export type PipelineTemplate = {
  id: string;
  name: string;
  parameters: Record<string, string>;
  pipeline: PipelineStep;
}

// Hook for GET /api/pipeline_templates
export function usePipelineTemplates() {
  return useQuery({
    queryKey: ["pipeline_templates"],
    queryFn: async (ctx) => {
      const response = await fetch(apiurl("pipeline/templates"), {
        signal: ctx.signal,
      });
      if (!response.ok) {
        throw new Error("Failed to fetch pipeline templates");
      }
      return response.json() as Promise<PipelineTemplate[]>;
    },
    staleTime: 1000 * 60 * 1, // 1 minute
  });
}