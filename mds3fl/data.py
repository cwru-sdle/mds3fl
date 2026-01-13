from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

def load_data(num_clients=5, batch_size=32):
    """Split MNIST dataset into N clients"""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    size = len(dataset) // num_clients
    subsets = random_split(dataset, [size]*num_clients)
    return [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
