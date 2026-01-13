import torch
import time
from federated_packet.network.tcp_client import TCPClient

if __name__ == "__main__":
    client = TCPClient(server_host="127.0.0.1", server_port=29500)
    fake_model = {
        "layer1.weight": torch.randn(2, 2),
        "layer1.bias": torch.randn(2)
    }

    print("[Client] Sending model to server and waiting for update...")
    updated_model = client.send_model_and_get_update(fake_model)

    print("[Client] Received global model:")
    for k, v in updated_model.items():
        print(k, v)