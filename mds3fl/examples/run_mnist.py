import argparse
import torch
from federated_packet.models.simple_cnn import SimpleCNN
from federated_packet.core import run
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def evaluate(model, device):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"\n🎯 Final Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Federated Packet Example")
    parser.add_argument("--algo", type=str, default="fedavg", choices=["fedavg", "fedprox", "fedadam"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    model = SimpleCNN()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained = run(
        model,
        algo=args.algo,
        num_clients=args.clients,
        rounds=args.rounds,
        epochs=args.epochs,
        device=device,
    )

    evaluate(trained, device)
