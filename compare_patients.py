import numpy as np
import os
import matplotlib.pyplot as plt

low_dir = "genAI/data/low_dose"
high_dir = "genAI/data/high_dose"
slice_index = 50  # Pick central slice

# Get all patient files
patients = sorted([f for f in os.listdir(low_dir) if f.endswith(".npy")])
num_patients = min(len(patients), 5)  # Limit to 5-6 patients for readability

plt.figure(figsize=(8, 2 * num_patients))

for i, fname in enumerate(patients[:num_patients]):
    low_path = os.path.join(low_dir, fname)
    high_path = os.path.join(high_dir, fname)

    low = np.load(low_path)[slice_index][..., 0]
    high = np.load(high_path)[slice_index][..., 0]

    # Normalize
    low = (low - np.min(low)) / (np.max(low) - np.min(low) + 1e-8)
    high = (high - np.min(high)) / (np.max(high) - np.min(high) + 1e-8)

    plt.subplot(num_patients, 2, i * 2 + 1)
    plt.imshow(low, cmap="gray")
    plt.title(f"Low-Dose: {fname[:30]}")
    plt.axis("off")

    plt.subplot(num_patients, 2, i * 2 + 2)
    plt.imshow(high, cmap="gray")
    plt.title(f"High-Dose: {fname[:30]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
