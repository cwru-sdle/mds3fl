import copy
from .server import Server
from .client import Client
from .data import load_data

def run(model, algo="fedavg", num_clients=5, rounds=3, epochs=1, device="cpu", **kwargs):
    """Main orchestrator for federated training"""
    dataloaders = load_data(num_clients)
    clients = [Client(i, copy.deepcopy(model), dataloaders[i], device) for i in range(num_clients)]
    server = Server(copy.deepcopy(model))

    for r in range(rounds):
        print(f"\n🌀 Round {r+1} using {algo}")
        client_states = []
        for client in clients:
            print(f"  ↳ Client {client.cid} local training...")
            state = client.local_train(epochs=epochs, lr=kwargs.get("client_lr", 0.01))
            client_states.append(state)
        server.aggregate(client_states, algo=algo, **kwargs)
        print(f"✅ Round {r+1} aggregation done.")

    return server.model
