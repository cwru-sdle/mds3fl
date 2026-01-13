from federated_packet.server import Server
from federated_packet.models.simple_cnn import SimpleCNN
import torch

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    server = Server(SimpleCNN())
    num_clients=5
    dataloaders = server.load_mnist(num_clients)
    server.init_clients(num_clients, SimpleCNN, dataloaders, device)
    server.run_federated(num_rounds=5, algo="fedavg", device=device)