import math
import time

import torch
import torch.multiprocessing as mp

from torch_mp import MasterSlaveCommunicator
from timeline_parser import parse_sim_export, compute_partition_shards
from timeline_utils import  chash, train_model, get_dataset_loaders

from lib.flcdata import FLCDataset, Model

#=== END OF IMPORTS ====

def worker_get_loaders(worker_id, device_comm, device, ds_folder, splt_per_partition, cpp):
    if worker_id == 0:
        train_dss = FLCDataset.LoadGroups(ds_folder=ds_folder, train=True)
        train_dss = [ds.to(device) for ds in train_dss]
        device_comm.send_broadcast(train_dss)
    else:
        train_dss = device_comm.recv_broadcast()

    return get_dataset_loaders(train_dss, splt_per_partition, cpp)

def worker(device_comm, world_comm, device_workers, worker_id, device_barrier, device_ch, n_masters, master_idx, masters_chs, in_ch, out_ch, device, ds_folder, splt_per_partition, cpp):
    IsMaster = worker_id == 0
    torch.cuda.set_device(device)
    
    in_size, out_size = FLCDataset.LoadSize(ds_folder)
    net = Model(insize=in_size, outsize=out_size)
    net.to(device)
    
    train_loaders = worker_get_loaders(worker_id, device_comm, device, ds_folder, splt_per_partition, cpp)
    print(f"Worker on device {device} {worker_id} ready with {len(train_loaders)} partitions.", flush=True)
    if IsMaster:
        out_ch.put("ready") # This is to signal Scheduler process that this worker is ready. to start receiving events.
 
    gmodel_map = {}
    local_upds = []
    curr_local_model = None
    
    while True: 
        event = in_ch.get()
        if event is None:
            break
        
        assert isinstance(event, dict), f"Event must be a dictionary, got {type(event)} : trimmed to {str(event)[:100]}..."
        
        if event['type'] == 'aggregation':
            version = event['version']
            if 'clean_up_models' in event and len(event['clean_up_models']) > 0:
                for v in event['clean_up_models']:
                    if v in gmodel_map:
                        del gmodel_map[v]
            
            if version != 1:
                _data = None if len(local_upds) == 0 else (curr_local_model, local_upds)
                if not IsMaster:
                    device_comm.send_gather(_data)
                else:
                    DC_UPDATES = device_comm.recv_gather(with_my_data=_data)
                del _data
                
             
            if IsMaster:
                metas = []
                if version == 1:
                    cpu_model = event['model']
                    model = {k: v.to(device) for k, v in cpu_model.items()}
                else:
                    updates = DC_UPDATES 
                            
                    if len(updates) == 0:
                        model = None
                        total_weight = 0
                    else:
                        assert isinstance(updates, list), f"Updates must be a list, got {type(updates)}"
                        assert all(isinstance(u, tuple) and len(u) == 2 for u in updates), f"Each update must be a tuple of (model, weight), got {[type(u) for u in updates]}"
                        assert all(isinstance(u[0], dict) for u in updates), f"Each model in updates must be a dictionary, got {[type(u[0]) for u in updates]}"
                        assert all(isinstance(u[1], list) and len(u[1]) > 0 for u in updates), f"Each update must have a non-empty list of contributors, got {[len(u[1]) for u in updates]}"
                        
                        UPD = [(model, sum([u['train_samples'] for u in metas])) for model, metas in updates]
                        total_weight = sum([n for _, n in UPD])
                        model = {k: sum([model[k] * n for model, n in UPD]) / total_weight for k in UPD[0][0].keys()}
                    
                    
                    CPU_MODEL = {k: v.cpu() for k, v in model.items()} if model is not None else None 
                    local_metas = [meta for _, meta in updates]
                    local_metas = [item for sublist in local_metas for item in sublist]  
                    others = world_comm.f_all_to_all(master_idx, (CPU_MODEL, local_metas))
                    
                    metas = [o[1] for o in others] #this is a list of lists of contributors, we need to flatten it
                    metas = [item for sublist in metas for item in sublist]  # flatten the list of lists
                    others = [{k: v.to(device) for k, v in o[0].items()} for o in others]
                    assert len(others) > 0, "No master had models to aggregate."
                    
                    weights = [m['train_samples'] for m in metas]
                    final_weight = sum(weights)
                    model = {k: sum([o[k] * w for o, w in zip(others, weights)]) / final_weight for k in others[0].keys()}

                device_comm.send_broadcast(model)
                
                if master_idx == 0:
                    CPU_MODEL = {k: v.cpu() for k, v in model.items()} if model is not None else None
                    out_ch.put((CPU_MODEL, {"contributors": metas}))
                    del CPU_MODEL

            else:
                model = device_comm.recv_broadcast()

            assert version not in gmodel_map, f"Model version {version} already exists."
            gmodel_map[version] = model
            continue
                

        version = event.get('version', -1)
        client = event.get('client', None)
        loader = train_loaders[client[0]][client[1]]
    
        assert "train_params" in event, "Train parameters not found in event."
        hyperparams = event['train_params']
        opt = hyperparams['optimizer']['type']
        assert opt == "sgd", f"Unsupported optimizer: {opt}. Only 'sgd' is supported."

        bs = hyperparams['batch_size']
        ephocs = hyperparams['ephocs']
        lr = hyperparams['optimizer']['learning_rate']
        momentum = hyperparams['optimizer'].get('momentum', 0.0)
        weight_decay = hyperparams['optimizer'].get('weight_decay', 0.0)

        loader.set_batch_size(bs)
        
        model = gmodel_map.get(version, None)
        assert model is not None, f"Model version {version} not found in global model map."
        
        # Copy the model (which is a state_dict and already on the device). copy it doing an indevice copy (no cpu transfer)
        model = {k: v.clone() for k, v in model.items()}
        meta, upd = train_model(net, model, loader, learning_rate=lr, ephocs=ephocs, momentum=momentum, weight_decay=weight_decay)
        
        meta['base_version'] = version
        meta['client'] = event['client']
        meta['batch_size'] = bs
        
        if curr_local_model is None:
            curr_local_model = {k: v * meta['train_samples'] for k, v in upd.items()}
        else:
            for k in curr_local_model.keys():
                curr_local_model[k] += upd[k] * meta['train_samples']
        
        local_upds.append(meta)

def repo_thread(repo, model_queue):
    while True:
        model = model_queue.get()
        if model is None:
            break

        repo.put_model(*model)

def run_simulation(sim, timeline, aggregations, ds_folder, repo_folder, args):
    assert len(aggregations) > 0, "At least one aggregation is required."
    assert len(timeline) > 0, "Timeline must contain at least one time step."
    assert len(sim["client_per_partition"]) == sim["npartitions"], "Number of partitions does not match number of client partitions."
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    
    from lib.flcdata import FLCDataset, Model
  
    
    logger.info("Loading datasets...")
    in_size, out_size = FLCDataset.LoadSize(ds_folder)
    splt_per_partition = compute_partition_shards(sim["client_per_partition"], sim.get("proportionalKnowledge", False))
    logger.info(f"Input size: {in_size}, Output size: {out_size}")
    logger.info(f"Timeline length: {len(timeline)}")
    logger.info(f"Aggregations: {len(aggregations)}")
    

    # Calculate max teorethical concurrency
    max_concurrency = 6
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
    
    logger.info(f"Max concurrency: {max_concurrency}")
    
    # check how many cuda devices are available
    n_devices = torch.cuda.device_count()
    max_worker_per_device = 4
    # worker_per_device = min(max_worker_per_device, max(1 math.ceil(max_concurrency / n_devices)))
    worker_per_device = min(max_worker_per_device, max(1, math.ceil(max_concurrency / n_devices)))
    
    
    
    devices = [torch.device(f"cuda:{i}") for i in range(n_devices)]
    assert n_devices > 0, "No CUDA devices available."
    logger.info(f"Number of CUDA devices: {n_devices}")
    logger.info(f"Worker per device: {worker_per_device}")
    
    ctx = mp.get_context('spawn')
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()
    device_barriers=        [ctx.Barrier(worker_per_device) for _ in range(n_devices)]
    device_inner_channel =  [ctx.Queue() for _ in range(n_devices)]
    masters_chs =           [ctx.Queue() for _ in range(n_devices)]
    device_comms =          [MasterSlaveCommunicator(ctx, worker_per_device) for _ in range(n_devices)]
    world_comm = MasterSlaveCommunicator(ctx, n_devices)
    
    processes = []
    for didx, device in enumerate(devices):
        
        for i in range(worker_per_device):
            args = (
                device_comms[didx],
                world_comm,
                worker_per_device, 
                i,
                device_barriers[didx], 
                device_inner_channel[didx],
                n_devices,
                didx,
                masters_chs,
                job_queue, 
                result_queue, 
                device, 
                ds_folder,
                splt_per_partition, 
                sim["client_per_partition"],
            )
            
            p = ctx.Process(target=worker, args=args)
            p.start()
            processes.append(p)
            
    logger.info(f"Started {len(processes)} worker processes.")
    
    # get ack from masters
    for _ in range(n_devices):
        _ = result_queue.get()
        
    logger.info("Workers initialized and ready.")

    # time.sleep(10000)
    tm_last_agg = time.perf_counter()
    
    
    base_model = Model(insize=in_size, outsize=out_size).state_dict()
    
    # spawn a thread to handle repository updates (normal python thread )
    from threading import Thread
    from queue import Queue
    from cmds.FLME.core.repository import ModelRepository
    repo_model_queue = Queue()
    repo = ModelRepository.from_disk(repo_folder, ignore_exists=False, window_size=2)
    repo_thread_instance = Thread(target=repo_thread, args=(repo, repo_model_queue), daemon=True)
    repo_thread_instance.start()
    del repo
    
    repo_model_queue.put((base_model, {
        "contributors": [],
    }))
        

    
    for _ in range(n_devices * worker_per_device):
        job_queue.put({
            "type": "aggregation",
            "version": 1,
            "model": base_model,
        })
        
    del base_model  
        
    _ = result_queue.get()

    logger.info("Base model pushed to repository and workers notified.")
    
    time.sleep(1)
    
    sparse_client_model_map = {}
    global_model_map = { 1: 0 }
    
    latest_model_version = 1
    pending_jobs = 0
    next_agg = aggregations[0]
    for t, events in enumerate(timeline):
        for event in events:
            ckey = chash(event['client'])
            if event['type'] == 'fetch':
                sparse_client_model_map[ckey] = latest_model_version
                global_model_map[latest_model_version] += 1
                
            elif event['type'] == 'train':
                pass
            
            elif event['type'] == 'send':        
                version = sparse_client_model_map[ckey]
                if version not in global_model_map:
                    raise ValueError(f"Client {ckey} not found in global model map.")
                
                del sparse_client_model_map[ckey]
                global_model_map[version] -= 1
                # if global_model_map[version]['ref-count'] == 0 and version != latest_model_version:
                #     del global_model_map[version]
                event['version'] = version
                job_queue.put(event)
                pending_jobs += 1
            else:
                raise ValueError(f"Unknown event type: {event['type']}")

        if next_agg == t:
            clean_up_models = []
            for version, ref_count in global_model_map.items():
                if ref_count == 0:
                    clean_up_models.append(version)
                    
            for version in clean_up_models:
                    del global_model_map[version]
                
            for _ in range(n_devices * worker_per_device):
                job_queue.put({
                    "type": "aggregation",
                    "version": latest_model_version + 1,
                    "model": None,  
                    "clean_up_models": clean_up_models,
                })

            global_model = result_queue.get()
            assert isinstance(global_model, tuple) and len(global_model) == 2, "Global model must be a tuple of (model, meta)."
            assert global_model[0] is not None, "Global model cannot be None."
            assert isinstance(global_model[0], dict), "Global model meta must be a dictionary."
            assert isinstance(global_model[1], dict), "Global model meta must be a dictionary."
            
            repo_model_queue.put(global_model)
            
            latest_model_version += 1
            global_model_map[latest_model_version] = 0
            now = time.perf_counter()
            
            logger.info(f"New GModel: {latest_model_version}, Elapsed Time: {now - tm_last_agg:.2f}s")
            tm_last_agg = now

            if latest_model_version > len(aggregations) - 1:
                logger.info("All aggregations completed.")
                break

            next_agg = aggregations[latest_model_version]

    logger.info("Simulation completed. Waiting for workers to finish...")
    for _ in range(len(processes)):
        job_queue.put(None)
        
    for p in processes:
        p.join()
        
        # join repo thread
    repo_model_queue.put(None)
    repo_thread_instance.join()
    
    logger.info("All workers finished.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation")
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
        print(f"MAIN: Error: {e}")
        import traceback
        traceback.print_exc()