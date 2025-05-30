import os
import numpy as np

low_dir = "genAI/data/low_dose"

for file in os.listdir(low_dir):
    if not file.endswith(".npy"):
        continue
    path = os.path.join(low_dir, file)
    data = np.load(path)
    for i, slice in enumerate(data):
        if slice.shape != (128, 128, 2):
            print(f"❌ {file}, slice {i} hatalı: shape = {slice.shape}")
