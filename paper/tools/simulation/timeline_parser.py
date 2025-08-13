import json
from typing import List, Tuple, Dict, Union, Any

EventType = str  # Literal["fetch", "train", "send"]
Client = Tuple[int, int]

EventFetchModel = Dict[str, Any]
EventTrainModel = Dict[str, Any]
EventSendModel = Dict[str, Any]

Event = Union[EventFetchModel, EventTrainModel, EventSendModel]
Timeline = List[List[Event]]

Simulation = Dict[str, Any]

SimExport = Dict[str, Any]

def parse_sim_export(json_file_path: str) -> SimExport:
    """
    Parses a JSON file with the structure of SIM_EXPORT (ITimeline).

    Args:
        json_file_path: The path to the JSON file.

    Returns:
        A dictionary representing the parsed SIM_EXPORT data.
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Basic validation
        if not isinstance(data, dict):
            raise ValueError("The JSON data must be a dictionary.")
        if "timeline" not in data:
            raise ValueError("The JSON data must contain a 'timeline' field.")
        if "aggregations" not in data:
            raise ValueError(
                "The JSON data must contain an 'aggregations' field."
            )
        if "sim" not in data:
            raise ValueError("The JSON data must contain a 'sim' field.")

        # Validation for aggregations
        if not isinstance(data["aggregations"], list):
            raise ValueError(
                "The 'aggregations' field must be a list of numbers."
            )
        for item in data["aggregations"]:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    "Each element in 'aggregations' must be a number."
                )

        # Validation for timeline (Timeline) - more complex
        timeline = data["timeline"]
        if not isinstance(timeline, list):
            raise ValueError("The 'timeline' field must be a list (Timeline).")

        for time_step in timeline:
            if not isinstance(time_step, list):
                raise ValueError(
                    "Each time step in the Timeline must be a list of events."
                )
            for event in time_step:
                if not isinstance(event, dict):
                    raise ValueError("Each event must be a dictionary.")
                if "type" not in event:
                    raise ValueError("Each event must have a 'type' field.")
                if "client" not in event:
                    raise ValueError("Each event must have a 'client' field.")

                # Further validation based on event type can be added here

        if not isinstance(data["sim"], dict):
            raise ValueError("The 'sim' field must be a dictionary.")
        
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValueError as e:
        raise ValueError(f"Error parsing JSON: {e}")

def compute_partition_shards(client_per_partition: List[int], proportional_knowledge: bool) -> List[int]:
    splt_per_partition = client_per_partition.copy()
    if proportional_knowledge:
        max_client_pp = max(splt_per_partition)
        for pidx, nclients in enumerate(client_per_partition):
            splt_per_partition[pidx] = max_client_pp
    return splt_per_partition


def max_timeline_concurrency(timeline: Timeline, aggregations: List[int]) -> int:
    max_concurrency = 1
    curr_max = 0
    agg_indx = 0
    for t, events in enumerate(timeline):
        for event in events:
            if event['type'] == 'send':
                curr_max += 1
        
        if t >= aggregations[agg_indx]:
            agg_indx += 1
            if curr_max > max_concurrency:
                max_concurrency = curr_max
            curr_max = 0
            
            if agg_indx >= len(aggregations):
                break

    return max_concurrency