import torch
import numpy as np
import matplotlib.pyplot as plt
from models.generator import UNetGenerator as fusion_generator
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

# === Ayarlar ===
generator_path = "genAI/generator_gan.pth"
low_path = "genAI/data/low_dose/patient_0001.npy"
high_path = "genAI/data/high_dose/patient_0001.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
slice_index = 30  # Ortalarda bir slice

# === Yardımcı: Normalize fonksiyonu ===
def normalize(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

# === Modeli yükle ===
model = fusion_generator().to(device)
model.load_state_dict(torch.load(generator_path, map_location=device))
model.eval()

# === Veriyi oku ===
low_volume = np.load(low_path)
high_volume = np.load(high_path)

print("✅ Değerlendirme başlatıldı...")
print(f"🔍 Low shape: {low_volume.shape} | High shape: {high_volume.shape}")

# === Slice seç
low_slice = normalize(low_volume[slice_index])
high_slice = normalize(high_volume[slice_index])

if np.mean(high_slice) < 0.005:
    print(f"⚠️ Slice {slice_index} çok karanlık (mean={np.mean(high_slice):.6f}), işlem atlandı.")
else:
    inp = torch.tensor(low_slice).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(inp).squeeze().cpu().numpy()

    out_img = normalize(out) * 255
    out_img = out_img.astype(np.uint8)
    high_img = (high_slice * 255).astype(np.uint8)
    low_img = (low_slice * 255).astype(np.uint8)

    # === PSNR & SSIM
    psnr = peak_signal_noise_ratio(high_img, out_img, data_range=255)
    ssim = structural_similarity(high_img, out_img, data_range=255)

    # === Görselleştirme
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(low_img, cmap='gray')
    plt.title("Low-Dose Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(out_img, cmap='gray')
    plt.title(f"Predicted\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(high_img, cmap='gray')
    plt.title("High-Dose Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"\n📊 PSNR: {psnr:.2f} dB")
    print(f"📊 SSIM: {ssim:.4f}")
