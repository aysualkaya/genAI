import numpy as np

low = np.load("genAI/data/low_dose/patient_0001.npy")
high = np.load("genAI/data/high_dose/patient_0001.npy")

print("Low min/max/mean:", low.min(), low.max(), low.mean())
print("High min/max/mean:", high.min(), high.max(), high.mean())
