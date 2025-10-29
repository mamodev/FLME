import numpy as np
import math
import os

SAVE_FOLDER = ""
def set_save_folder(folder):
    global SAVE_FOLDER
    os.makedirs(folder, exist_ok=True)
    SAVE_FOLDER = folder
    

def save_exp(exp, path = None):
    global SAVE_FOLDER
    import json
    
    p = os.path.join(SAVE_FOLDER, path)
    with open(p, 'w') as f:
        json.dump(exp, f)

    return path
    
def uniform_cpp(partitions, cpp):
    return [cpp] * partitions

def get_all_cid(cpp):
    ids = []
    for p in range(len(cpp)):
        for c in range(cpp[p]):
            ids.append((p, c))
    
    return ids

def add_constant_training_params(timeline, params, straggler_params_fact=None):
    straggler_params_fact = straggler_params_fact or (lambda: params)
    for events in timeline:
        for event in events:
            if event['type'] == 'send':
                if event.get('_stragg'):
                    event['train_params'] = straggler_params_fact()
                else:
                    event['train_params'] = params

def generate_sync_timeline(sim, cpa, straggler_filter=lambda t,c : True, allow_partial_part=False):
    assert cpa <= sum(sim['client_per_partition']), f"cpa must be less than the sum of cpp {sum(sim['client_per_partition'])} > {cpa}"
    events = []
    aggregations = []
    
    sf = straggler_filter
    
    all_clients = get_all_cid(sim['client_per_partition'])
    while len(aggregations) < sim['naggregations']:
        last_agg = len(aggregations)
        idx = np.random.choice(len(all_clients), size=cpa, replace=False)
        selected_clients = [all_clients[i] for i in idx]
        events.append([{'type': 'fetch', 'client': c} for c in selected_clients if sf(last_agg, c) or allow_partial_part])
        events.append([{'type': 'train', 'client': c} for c in selected_clients if sf(last_agg, c) or allow_partial_part])
        events.append([{'type': 'send', 'client': c, "__stragg": sf(last_agg, c)} for c in selected_clients  if sf(last_agg, c) or allow_partial_part])
        aggregations.append(aggregations[-1] + 3 if len(aggregations) > 0 else 2)
    
    return events, aggregations   

def random_straggler_filt_fact(nclients, min_drp_prc=0, max_drp_prc=1):
    assert min_drp_prc <= max_drp_prc, "min_drp_prc must be less than or equal to max_drp_prc"
    assert min_drp_prc >= 0 and max_drp_prc <= 1, "drp_prc must be in the range [0, 1]"
    
    tick_cache = {}
        
    def filter(t, c):
        if t not in tick_cache:
            tick_cache[t] = {}
            tick_cache[t]['count'] = math.ceil(np.random.uniform(min_drp_prc, max_drp_prc) * nclients)
            tick_cache[t]['clients'] = {}

        if c in tick_cache[t]['clients']:
            return False

        if len(tick_cache[t]['clients']) == tick_cache[t]['count']:
            return True
        else:
            tick_cache[t]['clients'][c] = True
            return False

    return filter

def fedprox_timeline(part=30, drp=0, lr=0.01, seed=42, cpa=10, mu=0, allow_partial_part=False, naggregations=200, epochs=20, batch_size=10):
    np.random.seed(seed)

    sim = {
        'npartitions': part,
        'naggregations': naggregations,
        'client_per_partition': uniform_cpp(part, 1),
        'proportionalKnowledge': False
    }

    tparams = {
        'ephocs': epochs,
        'batch_size': 10,
        'mu': mu,
        'optimizer': {  
            'type': 'sgd',
            'momentum': 0,
            'learning_rate': lr,
            'weight_decay': 0
        }
    }

    timeline, aggregations = generate_sync_timeline(
            sim, 
            cpa, 
            random_straggler_filt_fact(cpa, drp, drp), 
            allow_partial_part
        )
    
    np.random.seed(seed)
    add_constant_training_params(timeline, tparams, lambda: { **tparams,  "ephocs": np.random.randint(1, tparams['ephocs'] + 1) })

    # add_constant_training_params(timeline, tparams)

    return {
        'timeline': timeline,
        'aggregations': aggregations,
        'sim': sim,
    }    