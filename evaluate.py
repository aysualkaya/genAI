import torch
import numpy as np
import cv2
from models.simple_sr import SimpleSR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Görüntüleri oku
low = cv2.imread("data/low_dose/highdose_000.png", cv2.IMREAD_GRAYSCALE)
high = cv2.imread("data/high_dose/highdose_000.png", cv2.IMREAD_GRAYSCALE)

# Model yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSR().to(device)
model.load_state_dict(torch.load("simple_sr.pth", map_location=device))
model.eval()

# Giriş tensor oluştur
inp = torch.tensor(low).unsqueeze(0).unsqueeze(0).float().to(device)
with torch.no_grad():
    out = model(inp).squeeze().cpu().numpy()

# Görüntüyü clip et
out_clipped = np.clip(out, 0, 255).astype(np.uint8)

# Değerlendirme
psnr = peak_signal_noise_ratio(high, out_clipped, data_range=255)
ssim = structural_similarity(high, out_clipped, data_range=255)

print(f"📊 PSNR: {psnr:.2f} dB")
print(f"📊 SSIM: {ssim:.4f}")
