import torch
import numpy as np
import os
from models.generator import UNetGenerator as fusion_generator

generator_path = "genAI/generator_gan.pth"
# Yanlış
# low_path = "data/low_dose/patient_0001.npy"
# high_path = "data/high_dose/patient_0001.npy"

# Doğru olan:
low_path = "genAI/data/low_dose/patient_0001.npy"
high_path = "genAI/data/high_dose/patient_0001.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model yükle ===
model = fusion_generator().to(device)
model.load_state_dict(torch.load(generator_path, map_location=device))
model.eval()

# === Veriyi yükle ===
low = np.load(low_path) / 255.0
high = np.load(high_path) / 255.0

print("✅ Dosyalar yüklendi.")
print("Low shape:", low.shape)
print("High shape:", high.shape)

inp = torch.tensor(np.transpose(low, (2, 0, 1))).unsqueeze(0).float().to(device)

# === Tahmin üret ===
with torch.no_grad():
    out = model(inp).squeeze().cpu().numpy()

print("\n📦 Output shape:", out.shape)
print("✅ Output min:", np.nanmin(out))
print("✅ Output max:", np.nanmax(out))
print("❗ NaN var mı (output):", np.isnan(out).any())

# === GT kontrolü
high_img = high[:, :, 0]
print("\n📏 High image shape:", high_img.shape)
print("✅ High min:", np.nanmin(high_img))
print("✅ High max:", np.nanmax(high_img))
print("❗ NaN var mı (high):", np.isnan(high_img).any())
