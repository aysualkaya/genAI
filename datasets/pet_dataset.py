import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PETDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_data = []
        self.high_data = []

        filenames = sorted(os.listdir(low_dir))
        for filename in filenames:
            low_volume = np.load(os.path.join(low_dir, filename), allow_pickle=True)
            high_volume = np.load(os.path.join(high_dir, filename), allow_pickle=True)

            # Her slice'ı ayrı örnek olarak kaydet
            for i in range(low_volume.shape[0]):
                self.low_data.append(low_volume[i])
                self.high_data.append(high_volume[i])

    def __len__(self):
        return len(self.low_data)

    def __getitem__(self, idx):
        low = np.expand_dims(self.low_data[idx], axis=0)  # [1, H, W]
        high = np.expand_dims(self.high_data[idx], axis=0)

        return {
            'low': torch.tensor(low, dtype=torch.float32),
            'high': torch.tensor(high, dtype=torch.float32)
        }
