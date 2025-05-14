
print(f"running {__file__}", flush=True)

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import List, Tuple, Dict, Union, Any
import time
import threading
# Define Python equivalents for the TypeScript types

# Event Types
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


class MemoryDataLoader:
    def __init__(
        self, data, targets, batch_size=1024, shuffle=False,
        n_partitions=1, partition_idx=0
    ):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_partitions = n_partitions
        self.partition_idx = partition_idx

        # Calculate partition indices
        total_len = len(self.data)
        part_size = total_len // n_partitions
        remainder = total_len % n_partitions

        # Compute start and end indices for this partition
        self.start_idx = partition_idx * part_size + min(partition_idx, remainder)
        self.end_idx = self.start_idx + part_size
        if partition_idx < remainder:
            self.end_idx += 1

        # Slice the data for this partition
        self.partition_data = self.data[self.start_idx:self.end_idx]
        self.partition_targets = self.targets[self.start_idx:self.end_idx]

    def __iter__(self):
        indices = list(range(len(self.partition_data)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)
            yield (
                self.partition_data[batch_indices],
                self.partition_targets[batch_indices]
            )
            
    def __len__(self):
        return len(self.partition_data)



def dataset_to_device(dataset, device, batch_size=1024, shuffle=False, n_partitions=1, partition_idx=0):
    dataset.data = dataset.data.to(device)
    dataset.targets = dataset.targets.to(device)
    return MemoryDataLoader(dataset.data, dataset.targets, batch_size=batch_size, shuffle=shuffle)


import torch
import numpy as np
import torch.nn.functional as functional
from io import BytesIO

def chash(client):
    return f"{client[0]}_{client[1]}"

def deep_clone_sdict(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}

# def deep_clone_sdict(state_dict):
#     # check if device is != cpu
#     cpu_model = {}
#     for k, v in state_dict.items():
#         if isinstance(v, torch.Tensor):
#             cpu_model[k] = v.cpu().clone()
#         else:
#             raise ValueError(f"Unsupported type {type(v)} for key {k}")

#     iobuff = BytesIO()
#     torch.save(cpu_model, iobuff)
#     iobuff.seek(0)

#     cpu_model = torch.load(iobuff)
#     iobuff.close()

#     # move back to original device
#     for k, v in cpu_model.items():
#         if isinstance(v, torch.Tensor):
#             cpu_model[k] = v.to(state_dict[k].device)
#         else:
#             raise ValueError(f"Unsupported type {type(v)} for key {k}")
    
#     return cpu_model


def train_model(net, model, train_loader, learning_rate=0.01):
    import cmds.FLME.protocol.protocol as protocol

    model = deep_clone_sdict(model)
    net.load_state_dict(model)
    
    
    ephocs = 15
    lr = learning_rate
    momentum = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    meta = protocol.ModelMeta(
        momentum=momentum,
        learning_rate=lr,
        train_loss=0.1,
        test_loss=0.2,
        local_epoch=ephocs,
        train_samples=len(train_loader), 
    )
    
    net.train()
    for e in range(ephocs):
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    return meta, deep_clone_sdict(net.state_dict())

def aggregate_model(_mdl, updates):
    local_models = [
        (upd, meta.train_samples)
        for meta, upd in updates
    ]
    
    model = {k: sum([m[k] * n for m, n in local_models]) / sum([n for _, n in local_models]) for k in local_models[0][0].keys()}

    return model, {
            "contributors": [
                {
                    "meta": meta.to_dict(),
                }
                for meta, m in updates
            ]
        }

def lr_linear_interpolation_factory(min, max, num_steps):
    def lr_linear_interpolation(step):
        return min + (max - min) * (step / num_steps)
    
    return lr_linear_interpolation


def run_simulation(sim, timeline, aggregations, ds_folder, repo_folder, args):

    import types
    from lib.flcdata import FLCDataset, SimpleModel
    from cmds.FLME.core.repository import ModelRepository
    import logging
    import torch.utils.bottleneck as bottleneck


    

    lr_gen = lr_linear_interpolation_factory(
        args.lr_range[0],
        args.lr_range[1],
        len(aggregations) - 1
    )
   
    repo = ModelRepository.from_disk(repo_folder, ignore_exists=False, window_size=2)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)


    logger.info("Loading datasets...")
    in_size, out_size = FLCDataset.LoadSize(ds_folder)
    train_dss = FLCDataset.LoadGroups(ds_folder=ds_folder, train=True)
    if len(train_dss) != sim["npartitions"]:
        raise ValueError(f"Number of dataset partitions ({len(train_dss)}) does not match number of partitions ({sim['npartitions']}) of the timeline.")

    logger.info("Datasets loaded.")
    logger.info(f"Number of datasets: {len(train_dss)}")
    logger.info(f"Moving datasets to device...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    proportional_knowledge = "proportionalKnowledge" in sim and sim["proportionalKnowledge"]
    splt_per_partition = sim["client_per_partition"]
    if proportional_knowledge:
        max_client_pp = max(splt_per_partition)
        for pidx, nclients in enumerate(sim["client_per_partition"]):
            splt_per_partition[pidx] = max_client_pp

    train_loaders = [
        [
            dataset_to_device(
                partition_ds,
                device,
                batch_size=4096,
                shuffle=False,
                n_partitions=splt_per_partition[pidx],
                partition_idx=cidx,
            )
            for cidx in range(sim["client_per_partition"][pidx])
        ]
        for pidx, partition_ds in enumerate(train_dss)
    ]

    logger.info("Datasets moved to device.")
    assert len(aggregations) > 0, "At least one aggregation is required."

    next_agg = aggregations[0]

    repo.put_model(SimpleModel(
        insize=in_size,
        outsize=out_size,
    ).state_dict(), {
        "contributors": [],
    })


    sparse_client_model_map = {}

    logger.info("Global model initialized. Starting simulation...")
    logger.info(f"Timeline length: {len(timeline)}")
    logger.info(f"Aggregations: {len(aggregations)}")

    updates = []
    
    
    tm_last_agg = time.perf_counter()
    net = SimpleModel(
                insize=in_size,
                outsize=out_size,
            )
    
    net.to(device)
    
    global_model_map = {
        1: {
            "model": net.state_dict(),
            'ref-count': 0,
        }
    }
    
    latest_model_version = 1
    
    
    for t, events in enumerate(timeline):
        for event in events:
            ckey = chash(event['client'])
            if event['type'] == 'fetch':
                version = latest_model_version
                sparse_client_model_map[ckey] = version
                global_model_map[version]['ref-count'] += 1
                
            
            elif event['type'] == 'train':
                pass
            
            elif event['type'] == 'send':
                # tm_start = time.perf_counter()
        
                version = sparse_client_model_map[ckey]
                if version not in global_model_map:
                    raise ValueError(f"Client {ckey} not found in global model map.")
                
                global_model_map[version]['ref-count'] -= 1
                
                model = global_model_map[version]['model']
                data_loader = train_loaders[event['client'][0]][event['client'][1]]

                if global_model_map[version]['ref-count'] == 0 and version != latest_model_version:
                    del global_model_map[version]

                del sparse_client_model_map[ckey]
                
                upd = train_model(net, model, data_loader, learning_rate=lr_gen(t))
                updates.append(upd)
            else:
                raise ValueError(f"Unknown event type: {event['type']}")

        if next_agg == t:
            # global_model, meta = aggregate_model(global_model_map[latest_model_version]['model'], updates)
            global_model, meta = aggregate_model(None, updates)


            if global_model_map[latest_model_version]['ref-count'] == 0:
                del global_model_map[latest_model_version]

            repo.put_model(global_model, meta)

            latest_model_version += 1
            global_model_map[latest_model_version] = {
                "model": global_model,
                'ref-count': 0,
            }

            now = time.perf_counter()

            logger.info(f"New GModel: {latest_model_version}, Elapsed Time: {now - tm_last_agg:.2f}s")
            tm_last_agg = now
            updates.clear()

            if latest_model_version > len(aggregations) - 1:
                logger.info("All aggregations completed.")
                break

            next_agg = aggregations[latest_model_version]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--timeline", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--ds-folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--repo-folder", type=str, required=True, help="Path to the model repository folder")
    parser.add_argument("--nuke-repo", action="store_true", help="Delete the model repository before starting")
    
    parser.add_argument(
        "--lr-range",
        type=float,  # Apply float conversion to each argument
        nargs=2,     # Expect exactly 2 arguments following --lr-range
        default=[0.01, 0.01], # Default can be a list or tuple
        metavar=("LR_START", "LR_END"), # Optional: Improves help message
        help="Learning rate range (start end) for the simulation"
    )   

    
    parser.add_argument("--lr-interpolation", type=str, default="linear", help="Interpolation method for learning rate")
    parser.add_argument("--lr-end-interpolation", type=int, default=0, help="End interpolation for learning rate")

    args = parser.parse_args()

    if args.nuke_repo:
        import shutil
        import os
        if os.path.exists(args.repo_folder):
            shutil.rmtree(args.repo_folder)
            print(f"Deleted repository folder: {args.repo_folder}")
    
    try:
        sim_data = parse_sim_export(args.timeline)
        timeline = sim_data["timeline"]
        aggregations = sim_data["aggregations"]
        sim = sim_data["sim"]
        run_simulation(sim, timeline, aggregations, args.ds_folder, args.repo_folder, args)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")





