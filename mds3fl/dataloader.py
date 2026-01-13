import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ThermalDataset(Dataset):
    """
    PyTorch Dataset for thermal npy images.
    Each sample returns:
        img:  (1, H, W) float32 tensor
        label: float32 tensor
    """

    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load npy image
        img = np.load(img_path)        # (H, W)
        img = img[:, :, None]          # -> (H, W, 1)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (1, H, W)

        label = torch.tensor(label, dtype=torch.float32)
        return img, label


class ThermalDataLoader:
    """
    High-level loader:
      - Read Excel
      - Collect image paths
      - Construct PyTorch Dataset / DataLoader
    """

    def __init__(self, excel_path, base_dir):
        self.excel_path = excel_path
        self.base_dir = base_dir

        self.pb1_image_paths = []
        self.pb2_image_paths = []
        self.pb3_image_paths = []
        self.wr2_image_paths = []

        self.pb1_label = []
        self.pb2_label = []
        self.pb3_label = []
        self.wr2_label = []

    # -------------------------------------------------------------
    # Read Excel and collect labels
    # -------------------------------------------------------------
    def retrieve_data(self):
        xl_file = pd.ExcelFile(self.excel_path)
        dfs = {sheet: xl_file.parse(sheet) for sheet in xl_file.sheet_names}

        beta = dfs["Beta Fraction"]

        WR2 = beta.iloc[:, [0, 5]].iloc[6:1107]
        PB1 = beta.iloc[:, [7, 12]].iloc[6:1062]
        PB2 = beta.iloc[:, [14, 19]].iloc[6:1103]
        PB3 = beta.iloc[:, [21, 26]].iloc[6:861]

        self.pb1_label = list(PB1["Unnamed: 12"])
        self.pb2_label = list(PB2["Unnamed: 19"])
        self.pb3_label = list(PB3["Unnamed: 26"])
        self.wr2_label  = list(WR2["Unnamed: 5"])

        # ------------------------------
        # Collect image paths
        # ------------------------------
        def collect_paths(subdir):
            paths = []
            full = os.path.join(self.base_dir, subdir)
            for root, _, files in os.walk(full, topdown=False):
                for f in files:
                    paths.append(os.path.join(root, f))
            paths.sort()
            return paths

        self.pb1_image_paths = collect_paths("pb1")[2:-21]
        self.pb2_image_paths = collect_paths("pb2")[4:-2]
        self.pb3_image_paths = collect_paths("pb3")[2:-4]
        self.wr2_image_paths = collect_paths("wr2")[2:]

        return (
            self.pb1_image_paths,
            self.pb2_image_paths,
            self.pb3_image_paths,
            self.wr2_image_paths,
            self.pb1_label,
            self.pb2_label,
            self.pb3_label,
            self.wr2_label,
        )

    # -------------------------------------------------------------
    # Build PyTorch Dataset / DataLoader
    # -------------------------------------------------------------
    def build_dataset(self, img_paths, labels):
        return ThermalDataset(img_paths, labels)

    def build_dataloader(self, img_paths, labels, batch_size=32, shuffle=True):
        dataset = self.build_dataset(img_paths, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # -------------------------------------------------------------
    # Flexible builder: support single or multiple client datasets
    # -------------------------------------------------------------
    def build_dataloaders(self, img_paths, labels, batch_size=32, shuffle=True):
        """
        Supports:
          1. img_paths = [path1, path2, ...] → builds ONE DataLoader
          2. img_paths = [[paths_client1], [paths_client2], ...] → builds MULTIPLE DataLoaders

        labels should match the structure of img_paths.
        """
        # Case 2: multiple clients (list of lists)
        if len(img_paths) > 0 and isinstance(img_paths[0], list):
            dataloaders = []
            for p, l in zip(img_paths, labels):
                ds = ThermalDataset(p, l)
                dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
                dataloaders.append(dl)
            return dataloaders

        # Case 1: single dataset
        return DataLoader(
            ThermalDataset(img_paths, labels),
            batch_size=batch_size,
            shuffle=shuffle
        )

    # -------------------------------------------------------------
    # Split a single dataset into multiple clients
    # -------------------------------------------------------------
    def split_into_clients(self, img_paths, labels, num_clients):
        """
        Splits a single dataset (img_paths, labels) into `num_clients` subsets.

        Example:
            paths, labels = loader.pb1_image_paths, loader.pb1_label
            client_paths, client_labels = loader.split_into_clients(paths, labels, 4)
            # Then pass these to build_dataloaders()

        Returns:
            client_img_paths:  list of lists
            client_labels:      list of lists
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be >= 1")

        # Convert to numpy arrays for easy splitting
        paths_arr = np.array(img_paths)
        labels_arr = np.array(labels)

        # Use numpy array_split to evenly divide
        path_splits = np.array_split(paths_arr, num_clients)
        label_splits = np.array_split(labels_arr, num_clients)

        client_img_paths = [list(p) for p in path_splits]
        client_labels = [list(l) for l in label_splits]

        return client_img_paths, client_labels

    # -------------------------------------------------------------
    # Unified Multi-Client Builder
    # Supports:
    #   1) multiple directories → each directory = one client
    #   2) single directory → split into N clients
    # -------------------------------------------------------------
    def build_clients_from_paths(self, img_paths, labels, num_clients=None,
                                 batch_size=32, shuffle=True):
        """
        Parameters:
            img_paths:
                Case 1: list of lists
                    Example: [pb1_paths, pb2_paths, pb3_paths, wr2_paths]
                    → each sublist becomes one client.

                Case 2: single list
                    Example: pb1_paths
                    → split into `num_clients` clients.

            labels:
                Must match the structure of img_paths.

            num_clients:
                Only required when img_paths is a single list.

        Returns:
            List[DataLoader] for each client.
        """

        # ------------------------------
        # Case 1: multiple directories → each directory = one client
        # ------------------------------
        if len(img_paths) > 0 and isinstance(img_paths[0], list):
            client_dls = []
            for p, l in zip(img_paths, labels):
                ds = ThermalDataset(p, l)
                dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
                client_dls.append(dl)
            return client_dls

        # ------------------------------
        # Case 2: one directory → split into N clients
        # ------------------------------
        if num_clients is None:
            raise ValueError("num_clients must be provided when img_paths is a single list.")

        # split into lists
        client_paths, client_labels = self.split_into_clients(img_paths, labels, num_clients)

        # build dataloaders
        client_dls = []
        for p, l in zip(client_paths, client_labels):
            ds = ThermalDataset(p, l)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            client_dls.append(dl)

        return client_dls