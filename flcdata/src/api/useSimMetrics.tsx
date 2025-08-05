import { useQuery } from "@tanstack/react-query";
import { apiurl } from "./backend";

export type SimulationMetric = {
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
};

type SimulationMetrics = SimulationMetric[];

export function useSimMetrics(sim_name: string) {
  return useQuery({
    queryKey: ["sim-metrics", sim_name],
    queryFn: async (ctx) => {
      console.log("Fetching metrics for simulation:", sim_name);

      const response = await fetch(apiurl(`simulation-metrics/${sim_name}`), {
        signal: ctx.signal,
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      console.log("Response headers:", response.headers);

      const contentType = response.headers.get("content-type");
      console.log("Content type:", contentType);
      if (!contentType || !contentType.includes("application/json")) {
        console.error("Unexpected content type:", contentType);
        throw new Error("Response is not JSON");
      }

      console.log("Response JSON being parsed...");
      try {
        const txt = await response.text();
        // console.log("Response text:", txt);
        // replace all NaN with 0
        const sanitizedTxt = txt.replace(/NaN/g, "0");

        const res = JSON.parse(sanitizedTxt);
        return res as SimulationMetrics;
      } catch (error) {
        console.error("Error parsing JSON:", error);
        throw new Error("Failed to parse JSON response");
      }
    },
    staleTime: 1000 * 60 * 5,
  });
}
