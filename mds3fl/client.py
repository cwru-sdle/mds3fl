import torch
from torch import nn
from torch.utils.data import DataLoader

class Client:
    """Simulated FL client"""
    def __init__(self, cid, model, dataloader, device="cpu"):
        self.cid = cid
        self.model = model
        self.data = dataloader
        self.device = device

    def set_dataloader(self, dataloader):
        self.data = dataloader

    def local_train(self, epochs=1, lr=0.01):
        """Perform local SGD training"""
        self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(epochs):
            for x, y in self.data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                optimizer.step()

        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def run_round(self, epochs=1, lr=0.01):
        print(f"[Client {self.cid}] Running local training...")
        return self.local_train(epochs=epochs, lr=lr)
