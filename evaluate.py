import torch
import numpy as np
from models.simple_sr import SimpleSR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
import matplotlib.pyplot as plt

# 🔹 1. Verileri yükle
low = np.load("genAI/data/low_dose/patient_0001.npy")
high = np.load("genAI/data/high_dose/patient_0001.npy")

# 🔹 2. İlk slice'ı ve sadece PET kanalını al
low_slice = low[0]  # (128, 128, 2)
high_slice = high[0]

low_pet = low_slice[..., 0] if low_slice.ndim == 3 else low_slice
high_pet = high_slice[..., 0] if high_slice.ndim == 3 else high_slice

# 🔹 3. Normalize et
low_pet = np.clip(low_pet, 0, 1)
high_pet = np.clip(high_pet, 0, 1)

# 🔹 4. Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSR().to(device)
model.load_state_dict(torch.load("simple_sr.pth", map_location=device))
model.eval()

# 🔹 5. Giriş tensorü oluştur ve tahmin yap
inp = torch.tensor(low_pet).unsqueeze(0).unsqueeze(0).float().to(device)
with torch.no_grad():
    pred = model(inp).squeeze().cpu().numpy()

# 🔹 6. Boyut uyumsuzluğu varsa yeniden boyutlandır
if pred.shape != high_pet.shape:
    print(f"⚠️ Boyut farkı: pred={pred.shape}, high_pet={high_pet.shape} → yeniden boyutlandırılıyor.")
    pred = resize(pred, high_pet.shape, preserve_range=True, anti_aliasing=True)

# 🔹 7. PSNR ve SSIM hesapla
psnr = peak_signal_noise_ratio(high_pet, pred, data_range=1.0)
ssim = structural_similarity(high_pet, pred, data_range=1.0)

# 🔹 8. Sonuçları yazdır
print(f"📊 PSNR: {psnr:.2f} dB")
print(f"📊 SSIM: {ssim:.4f}")

# 🔹 9. Görsel karşılaştırma
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(low_pet, cmap="gray")
plt.title("Low-Dose Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred, cmap="gray")
plt.title(f"Predicted\nPSNR: {psnr:.2f} | SSIM: {ssim:.4f}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(high_pet, cmap="gray")
plt.title("High-Dose Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()
with torch.no_grad():
    pred = model(inp)
    pred = torch.sigmoid(pred)  # Sigmoid eklemezsek görüntü tamamen beyaz olabilir
    pred = pred.squeeze().cpu().numpy()

pred = np.clip(pred, 0, 1)
