# MDS3FL

MDS3FL is a small PyTorch-based federated learning package. 


## Features

- Simulated federated clients with local SGD training.
- Aggregation algorithms:
  - FedAvg
  - FedAdam
  - FedProx-style server update
  - FedDyn server-side update
- Simple CNN model for MNIST experiments.
- MNIST loader that splits training data across clients.
- Thermal image dataset loader for `.npy` files with labels from Excel sheets.
- TCP client/server utilities for round-based model exchange.

## Project Layout

```text
mds3fl/
  algorithm.py              # FedAvg, FedAdam, FedProx, FedDyn aggregation logic
  client.py                 # Simulated federated learning client
  core.py                   # High-level training entry point, currently needs Server implementation
  data.py                   # MNIST loading and client splitting
  dataloader.py             # Thermal .npy image dataset and loader helpers
  models/
    simple_cnn.py           # Simple CNN for MNIST
  network/
    tcp_client.py           # TCP client for sending local model states
    tcp_server.py           # TCP server for collecting and aggregating client states
  examples/
    TCP_test.py             # End-to-end threaded TCP demo
    run_mnist.py            # Legacy MNIST script, currently blocked by core.py
    run_federated_training.py
    test_client.py
    test_server.py
```

## Requirements

The package metadata currently declares:

```text
torch
numpy
```

Some modules and examples also require:

```text
torchvision
pandas
openpyxl
```

Use a Python environment with Python 3.8 or newer.

## Installation

From the project root:

```bash
python -m pip install -e .
```

For the examples and data loaders:

```bash
python -m pip install torch torchvision numpy pandas openpyxl
```

If you are installing PyTorch manually, choose the command that matches your
platform and CUDA version from the official PyTorch installation guide.

## Core Concepts

### Client

`mds3fl.client.Client` wraps a local model and a PyTorch `DataLoader`. Each
client trains its own copy of the model and returns the trained model
parameters.

```python
from mds3fl.client import Client

client = Client(
    cid=0,
    model=model,
    dataloader=train_loader,
    device="cpu",
)

local_state = client.local_train(epochs=1, lr=0.01)
```

### Aggregation

Aggregation functions operate on lists of PyTorch model `state_dict` objects.

```python
from mds3fl.algorithm import fedavg

global_state = fedavg([client_state_1, client_state_2])
model.load_state_dict(global_state)
```

Available functions:

- `fedavg(client_states)`
- `fedadam(client_states, global_model_state, m_t, v_t, ...)`
- `fedprox(client_states, global_model_state, mu=0.01)`
- `feddyn_server_step(client_states, w_t, g_t, alpha)`

### Model

The included `SimpleCNN` is designed for MNIST-shaped inputs:

```python
from mds3fl.models.simple_cnn import SimpleCNN

model = SimpleCNN()
```

Input shape:

```text
[batch_size, 1, 28, 28]
```

Output shape:

```text
[batch_size, 10]
```

## MNIST Data Loading

`mds3fl.data.load_data` downloads MNIST into `./data` and splits the training
set evenly across clients.

```python
from mds3fl.data import load_data

dataloaders = load_data(num_clients=5, batch_size=32)
```

Each item in `dataloaders` is a PyTorch `DataLoader` for one client.

## TCP Federated Learning Demo

The TCP implementation uses a simple length-prefixed pickle protocol:

```text
Client -> Server: local model state_dict
Server -> Client: aggregated model state_dict
```

Because the transport uses `pickle`, it should only be used with trusted local
clients or controlled research environments. Do not expose this protocol to
untrusted networks.

The most complete runnable example is:

```bash
python -m mds3fl.examples.TCP_test
```

This script:

1. Starts a local TCP server.
2. Splits MNIST across multiple client threads.
3. Trains each client locally.
4. Sends local model states to the server.
5. Aggregates models on the server.
6. Sends the same aggregated model back to all clients.
7. Evaluates each client model after every round.

You can change the experiment settings near the top of
`mds3fl/examples/TCP_test.py`:

```python
HOST = "127.0.0.1"
PORT = 5000
NUM_CLIENTS = 2
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
ALGO = "fedavg"   # "fedavg" | "fedprox" | "feddyn"
```

## TCP API

### Server

```python
from mds3fl.network.tcp_server import TCPServer

server = TCPServer(
    model=global_model,
    algo="fedavg",
    networkstatus=True,
    host="127.0.0.1",
    port=5000,
    max_clients=2,
)

aggregated_state = server.run_round(timeout=300)
```

### Client

```python
from mds3fl.network.tcp_client import TCPClient

client = TCPClient(
    server_host="127.0.0.1",
    server_port=5000,
    algo="fedavg",
)

new_state = client.run_round(model.state_dict())
model.load_state_dict(new_state)
```

## Thermal Dataset Loader

`mds3fl.dataloader` contains helpers for thermal image datasets stored as `.npy`
arrays, with labels read from an Excel file.

```python
from mds3fl.dataloader import ThermalDataLoader

loader = ThermalDataLoader(
    excel_path="path/to/labels.xlsx",
    base_dir="path/to/image/root",
)

(
    pb1_paths,
    pb2_paths,
    pb3_paths,
    wr2_paths,
    pb1_labels,
    pb2_labels,
    pb3_labels,
    wr2_labels,
) = loader.retrieve_data()

client_loaders = loader.build_clients_from_paths(
    img_paths=[pb1_paths, pb2_paths, pb3_paths, wr2_paths],
    labels=[pb1_labels, pb2_labels, pb3_labels, wr2_labels],
    batch_size=32,
    shuffle=True,
)
```

The resulting dataset returns:

```text
image: torch.float32 tensor with shape [1, H, W]
label: torch.float32 tensor
```

## Current Code Status

Several files are active and internally consistent:

- `mds3fl/client.py`
- `mds3fl/algorithm.py`
- `mds3fl/data.py`
- `mds3fl/dataloader.py`
- `mds3fl/models/simple_cnn.py`
- `mds3fl/network/tcp_client.py`
- `mds3fl/network/tcp_server.py`
- `mds3fl/examples/TCP_test.py`

Some older entry points reference modules or methods that are not present in the
current package layout:

- `mds3fl/core.py` imports `mds3fl.server.Server`, but `server.py` is not
  included.
- `mds3fl/examples/run_mnist.py` imports `mds3fl.core.run`, so it is blocked by
  the missing `Server` implementation.
- `mds3fl/examples/run_federated_training.py` also depends on
  `mds3fl.server.Server`.
- `mds3fl/examples/test_server.py` imports `mds3fl.tcp_server`, but the current
  path is `mds3fl.network.tcp_server`.
- `mds3fl/examples/test_client.py` calls `send_model_and_get_update`, but the
  current TCP client method is `run_round`.

Until those files are updated, prefer `mds3fl/examples/TCP_test.py` for a full
end-to-end demo.

## Development Notes

Run a syntax check:

```bash
python -m compileall -q mds3fl
```

Suggested next cleanup tasks:

1. Add or remove the missing `Server` abstraction used by `core.py`.
2. Update the older example scripts to use `mds3fl.network.tcp_server` and
   `TCPClient.run_round`.
3. Add missing runtime dependencies to `pyproject.toml`.
4. Add tests for aggregation functions and TCP round behavior.
5. Decide whether FedProx should include the proximal term during client-side
   local training instead of only adjusting the server aggregation result.

## License
This project is licensed under the BSD 3-Clause License.


