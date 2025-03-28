import argparse 
import asyncio

from core.connection import Connection
from core.repository import ModelRepository

import protocol.protocol as protocol

import asyncio
import time  

async def strategy(s, n):
    while True:
        await asyncio.sleep(0)
        
        start_time = time.time()
        
        while len(s.listeners) < n:
            await s.listeners.on_size_change()
        
        count = await Connection.broadcast_event(protocol.TrainEventID, s.listeners.extract())
        assert count == n, f"Expected {n} clients, got {count}, some clients may have disconnected"

        while len(s.updates) < count:
            assert len(s.listeners) == n, f"Expected {n} clients, got {len(s.listeners)}, some clients may have disconnected, while waiting for updates"
            await s.updates.on_size_change()

        local_models = s.updates.extract()
        s.updates.clear()

        local_models = [(p.data.model_state, p.meta.train_samples) for p in local_models]
        
        model = {k: sum([m[k] * n for m, n in local_models]) / sum([n for _, n in local_models]) for k in local_models[0][0].keys()}
        
        s.repo.put_model(model)

        count = await Connection.broadcast_event(protocol.NewGlobalModelEventID, s.listeners.extract())
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training round completed in {elapsed_time:.2f} seconds.")


def fed_avg_factory(arga):
    async def fed_avg(s):
        await strategy(s, args.nclients)
    return fed_avg

def fed_prox_factory(args):
    raise NotImplementedError()

STRAT_FACT = {
    "fed_avg": fed_avg_factory,
    "fed_prox": fed_prox_factory
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run server")
    parser.add_argument("--host", type=str, help="Host to bind to", default="localhost")
    parser.add_argument("--port", type=int, help="Port to bind to", default=8888)
    parser.add_argument("--repo", type=str, help="Path to model repository", default="./repo")

    subparsers = parser.add_subparsers(dest="strategy", help="Federated learning strategy")

    fed_avg_parser = subparsers.add_parser("fed_avg", help="Federated Averaging strategy")
    fed_avg_parser.add_argument("--nclients", type=int, required=True, help="Number of clients")

    fed_prox_parser = subparsers.add_parser("fed_prox", help="Federated Proximal strategy")
    fed_prox_parser.add_argument("--nclients", type=int, required=True, help="Number of clients")
    fed_prox_parser.add_argument("--mu", type=float, required=True, help="Regularization parameter for FedProx")
    
    args = parser.parse_args()
    if args.strategy is None:
        parser.error("the following argument is required: --strategy")

    assert args.strategy in STRAT_FACT, f"Invalid strategy: {args.strategy}"
    strat = STRAT_FACT[args.strategy](args)

    from core.server import Server
    repo = ModelRepository.from_disk(args.repo, ignore_exists=True, window_size=10)

    from model import Model
    repo.put_model(Model().state_dict())

    server = Server(args.host, args.port, strat, repo)
    asyncio.run(server.run())