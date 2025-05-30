import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_samples(high_dose_dir="genAI/data/high_dose", low_dose_dir="genAI/data/low_dose", num_samples=5):
    high_dose_files = sorted([f for f in os.listdir(high_dose_dir) if f.endswith(".npy")])[:num_samples]
    low_dose_files = sorted([f for f in os.listdir(low_dose_dir) if f.endswith(".npy")])[:num_samples]

    plt.figure(figsize=(10, num_samples * 3))
    for i in range(num_samples):
        high_volume = np.load(os.path.join(high_dose_dir, high_dose_files[i]))
        low_volume = np.load(os.path.join(low_dose_dir, low_dose_files[i]))

        slice_idx = high_volume.shape[0] // 2

        def get_pet(volume):
            if volume.ndim == 4:  # (D, H, W, 2)
                return volume[slice_idx, :, :, 0]
            elif volume.ndim == 3:
                if volume.shape[-1] == 2:
                    return volume[slice_idx, :, 0]
                else:
                    return volume[slice_idx]
            else:
                raise ValueError(f"Unexpected shape: {volume.shape}")

        low_slice = get_pet(low_volume)
        high_slice = get_pet(high_volume)

        # Normalize for visibility
        def normalize(img):
            img = np.nan_to_num(img)  # remove NaNs if any
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        low_slice = normalize(low_slice)
        high_slice = normalize(high_slice)

        # Plotting
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(low_slice, cmap="gray")
        plt.title(f"Low-Dose: {low_dose_files[i][:35]}...", fontsize=8)
        plt.axis("off")

        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(high_slice, cmap="gray")
        plt.title(f"High-Dose: {high_dose_files[i][:35]}...", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_samples()
