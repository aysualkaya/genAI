import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from models.model_il import ModelILGenerator

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# === Custom Target (NO BaseTarget) ===
class GeneratorOutputTarget:
    def __init__(self, target_type="mean"):
        self.target_type = target_type

    def __call__(self, model_output):
        if self.target_type == "mean":
            return model_output.mean()
        elif self.target_type == "max":
            return model_output.max()
        elif self.target_type == "sum":
            return model_output.sum()
        elif self.target_type == "center":
            h, w = model_output.shape[-2:]
            center_h, center_w = h // 4, w // 4
            center_region = model_output[..., center_h:3*center_h, center_w:3*center_w]
            return center_region.mean()
        else:
            return model_output.mean()

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelILGenerator(in_channels=1).to(device)
model.load_state_dict(torch.load("genAI/generator_gan.pth", map_location=device))
model.eval()

low_dir = "genAI/data/low_dose"
high_dir = "genAI/data/high_dose"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

filenames = sorted(os.listdir(low_dir))[:5]  # Dosya sayısını arttırabilirsin

fig, axes = plt.subplots(len(filenames), 3, figsize=(12, len(filenames) * 3))

for i, fname in enumerate(filenames):
    low_path = os.path.join(low_dir, fname)
    high_path = os.path.join(high_dir, fname)

    try:
        low = np.load(low_path)[50]
        high = np.load(high_path)[50]
    except Exception as e:
        print(f"⚠️ Skipping {fname} due to error: {e}")
        continue

    low_pet = low if low.ndim == 2 else low[..., 0]
    high_pet = high if high.ndim == 2 else high[..., 0]

    # Normalize for overlay
    low_norm = (low_pet - low_pet.min()) / (low_pet.max() - low_pet.min() + 1e-8)
    low_rgb = np.repeat(low_norm[..., np.newaxis], 3, axis=-1)

    # Input tensor
    inp = torch.tensor(low_pet).unsqueeze(0).unsqueeze(0).float().to(device)

    # Grad-CAM++
    target_layers = [model.bottleneck]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    target = GeneratorOutputTarget(target_type="mean")  # Diğer seçenekler: "max", "center", "sum"

    grayscale_cam = cam(input_tensor=inp, targets=[target])[0]

    # Blur and normalize
    grayscale_cam = cv2.GaussianBlur(grayscale_cam, (5, 5), sigmaX=1)
    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    overlay_image = show_cam_on_image(low_rgb, grayscale_cam, use_rgb=True)

    # Plotting
    axes[i, 0].imshow(low_pet, cmap="gray")
    axes[i, 0].set_title(f"Low: {fname[:40]}", fontsize=8)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(overlay_image)
    axes[i, 1].set_title("Grad-CAM++ Overlay", fontsize=8)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(high_pet, cmap="gray")
    axes[i, 2].set_title(f"High: {fname[:40]}", fontsize=8)
    axes[i, 2].axis("off")

    # Save overlay image
    save_path = os.path.join(output_dir, f"overlay_{fname.replace('.npy', '')}.png")
    cv2.imwrite(save_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

plt.tight_layout()
plt.show()
print("✅ Grad-CAM++ visualization completed.")
