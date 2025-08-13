
import time
import logging
import torch
import torch.nn.functional as functional

from timeline_parser import parse_sim_export, compute_partition_shards
from timeline_utils import get_dataset_loaders, chash, deep_clone_sdict, train_model

from flcdata import FLCDataset
from repository import ModelRepository
from nets import MODELS

def aggregate_model(_mdl, updates):
    local_models = [
        (upd, meta['train_samples'])
        for meta, upd in updates
    ]
    
    model = {k: sum([m[k] * n for m, n in local_models]) / sum([n for _, n in local_models]) for k in local_models[0][0].keys()}

    return model, {
            "contributors": [
                meta for meta, model in updates
            ]
        }

def run_simulation(sim, timeline, aggregations, ds_folder, repo_folder, args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model = MODELS[args.net]
    repo = ModelRepository.from_disk(repo_folder, ignore_exists=False, window_size=2)

    logger.info("Loading datasets...")
    in_size, out_size = FLCDataset.LoadSize(ds_folder)
    train_dss = FLCDataset.LoadGroups(ds_folder=ds_folder, train=True)
    if len(train_dss) != sim["npartitions"]:
        raise ValueError(f"Number of dataset partitions ({len(train_dss)}) does not match number of partitions ({sim['npartitions']}) of the timeline.")

    train_dss = [ds.to(device) for ds in train_dss]

    logger.info("Datasets loaded.")
    logger.info(f"Number of datasets: {len(train_dss)}")
    logger.info(f"Moving datasets to device...")
    
    splt_per_partition = compute_partition_shards(sim["client_per_partition"], sim.get("proportionalKnowledge", False))


    train_loaders = get_dataset_loaders(train_dss, splt_per_partition, sim["client_per_partition"])

    logger.info("Datasets moved to device.")
    assert len(aggregations) > 0, "At least one aggregation is required."

    next_agg = aggregations[0]

    repo.put_model(Model(
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
    net = Model(
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


                assert "train_params" in event, "Train parameters not found in event."
                hyperparams = event['train_params']
                opt = hyperparams['optimizer']['type']
                assert opt == "sgd", f"Unsupported optimizer: {opt}. Only 'sgd' is supported."

                bs = hyperparams['batch_size']
                ephocs = hyperparams['ephocs']
                lr = hyperparams['optimizer']['learning_rate']
                momentum = hyperparams['optimizer'].get('momentum', 0.0)
                weight_decay = hyperparams['optimizer'].get('weight_decay', 0.0)

                data_loader.set_batch_size(bs)

                model = deep_clone_sdict(net.state_dict())
                meta, upd = train_model(net, model, data_loader, learning_rate=lr, ephocs=ephocs, momentum=momentum, weight_decay=weight_decay)

                meta['base_version'] = version
                meta['client'] = event['client']
                meta['batch_size'] = bs

                updates.append((meta, upd))
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
    
    models_options = list(MODELS.keys())

    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--net", type=str, choices=models_options, required=True, help="Network architecture to use")
    parser.add_argument("--timeline", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--ds-folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--repo-folder", type=str, required=True, help="Path to the model repository folder")
    parser.add_argument("--nuke-repo", action="store_true", help="Delete the model repository before starting")
    
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





