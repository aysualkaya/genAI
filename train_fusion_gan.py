import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.model_il import ModelILGenerator

# === Basit PatchGAN Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# === Dataset Loader ===
class PETCTDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.files = sorted([f for f in os.listdir(low_dir) if f.endswith(".npy")])

        self.slice_counts = []
        for f in self.files:
            arr = np.load(os.path.join(self.low_dir, f))
            self.slice_counts.append(arr.shape[0])

        print(f"📦 Toplam dosya: {len(self.files)}")
        print(f"🧶 Toplam slice: {sum(self.slice_counts)}")

    def __len__(self):
        return sum(self.slice_counts)

    def __getitem__(self, idx):
        running = 0
        for vol_idx, count in enumerate(self.slice_counts):
            if idx < running + count:
                slice_idx = idx - running
                file = self.files[vol_idx]

                low_vol = np.load(os.path.join(self.low_dir, file))
                high_vol = np.load(os.path.join(self.high_dir, file))

                low = low_vol[slice_idx]  # (H, W, 2)
                high = high_vol[slice_idx]  # (H, W) veya (H, W, 1)

                # Eğer high tek kanallıysa, 2 kanallı yap
                if len(high.shape) == 2:
                    high = np.stack([high, high], axis=-1)  # (H, W, 2)
                elif high.shape[-1] == 1:
                    high = np.concatenate([high, high], axis=-1)  # (H, W, 2)

                if low.shape[-1] != 2 or high.shape[-1] != 2:
                    raise ValueError(f"Beklenmeyen kanal sayısı: low={low.shape}, high={high.shape}")

                low = np.transpose(low, (2, 0, 1))  # (2, H, W)
                high = np.transpose(high, (2, 0, 1))  # (2, H, W)

                return torch.tensor(low, dtype=torch.float32), torch.tensor(high[0:1], dtype=torch.float32)
            running += count

# === Ayarlar ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = ModelILGenerator().to(device)
discriminator = Discriminator().to(device)

loss_fn = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
opt_g = optim.Adam(generator.parameters(), lr=1e-4)
opt_d = optim.Adam(discriminator.parameters(), lr=1e-4)

dataloader = DataLoader(PETCTDataset("genAI/data/low_dose", "genAI/data/high_dose"), batch_size=4, shuffle=True)

# === Eğitim Döngüsü ===
epochs = 50
for epoch in range(epochs):
    g_loss_total, d_loss_total = 0, 0
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Discriminator
        real_pred = discriminator(y)
        fake_y = generator(x).detach()
        fake_pred = discriminator(fake_y)

        loss_d = bce(real_pred, torch.ones_like(real_pred)) + \
                 bce(fake_pred, torch.zeros_like(fake_pred))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Generator
        gen_y = generator(x)
        adv = discriminator(gen_y)
        loss_g = loss_fn(gen_y, y) + 0.001 * bce(adv, torch.ones_like(adv))
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        g_loss_total += loss_g.item()
        d_loss_total += loss_d.item()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss G: {g_loss_total/i:.4f} | Loss D: {d_loss_total/i:.4f}")

# === Modeli Kaydet ===
torch.save(generator.state_dict(), "genAI/generator_modelil.pth")
print("\u2705 Model kaydedildi.")