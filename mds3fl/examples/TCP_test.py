import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mds3fl.network.tcp_server import TCPServer
from mds3fl.network.tcp_client import TCPClient

# =====================
# Config
# =====================
HOST = "127.0.0.1"
PORT = 5000
NUM_CLIENTS = 2
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
ALGO = "fedavg"   # "fedavg" | "fedprox" | "feddyn"

DEVICE = "cpu"

# =====================
# Simple CNN for MNIST
# =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        # MNIST: input 28x28 -> conv1(3x3) -> 26x26 -> conv2(3x3) -> 24x24 -> maxpool(2) -> 12x12
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # Debug safety: ensure feature dimension matches fc1
        # Expected: [batch_size, 32*12*12]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# =====================
# Local training (pure local SGD)
# =====================
def local_train(model, dataloader, epochs=1):
    model.train()
    opt = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

# =====================
# Evaluation
# =====================
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

# =====================
# Client thread
# =====================
def client_thread_fn(client_id, dataloader, test_loader):
    client = TCPClient(HOST, PORT, algo=ALGO)
    model = SimpleCNN().to(DEVICE)

    for r in range(ROUNDS):
        # Local training
        local_train(model, dataloader, epochs=LOCAL_EPOCHS)

        # Send model to server, receive aggregated model
        new_state = client.run_round(model.state_dict())
        model.load_state_dict(new_state)

        acc = evaluate(model, test_loader)
        print(f"[Client {client_id}] Finished round {r} | Test Acc: {acc:.4f}")

# =====================
# Main
# =====================
def main():
    # Prepare MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Split dataset into 2 parts (simple IID split)
    indices = torch.randperm(len(dataset))
    split = len(dataset) // NUM_CLIENTS
    subsets = [
        Subset(dataset, indices[i * split:(i + 1) * split])
        for i in range(NUM_CLIENTS)
    ]

    loaders = [
        DataLoader(subsets[i], batch_size=BATCH_SIZE, shuffle=True)
        for i in range(NUM_CLIENTS)
    ]

    # Start server
    global_model = SimpleCNN().to(DEVICE)
    server = TCPServer(
        global_model,
        algo=ALGO,
        alpha=0.1,
        networkstatus=True,
        host=HOST,
        port=PORT,
        max_clients=NUM_CLIENTS,
    )

    def server_loop():
        r = 0
        while True:
            print(f"\n[Server] === Waiting for client batch (round {r}) ===")
            server.run_round(timeout=300)
            r += 1

    server_thread = threading.Thread(target=server_loop, daemon=True)
    server_thread.start()

    # Give server time to start listening
    time.sleep(1)

    # Start clients
    client_threads = []
    for i in range(NUM_CLIENTS):
        t = threading.Thread(target=client_thread_fn, args=(i, loaders[i], test_loader))
        t.start()
        client_threads.append(t)

    for t in client_threads:
        t.join()

    print("\nDemo finished.")


if __name__ == "__main__":
    main()