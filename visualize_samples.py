import matplotlib.pyplot as plt
import cv2
import os

def visualize_samples(high_dose_dir="data/high_dose", low_dose_dir="data/low_dose", num_samples=5):
    high_dose_files = sorted(os.listdir(high_dose_dir))[:num_samples]
    low_dose_files = sorted(os.listdir(low_dose_dir))[:num_samples]
    
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # High-Dose görüntü
        high_img = cv2.imread(os.path.join(high_dose_dir, high_dose_files[i]), cv2.IMREAD_GRAYSCALE)
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(high_img, cmap="gray")
        plt.title(f"High-Dose: {high_dose_files[i]}")
        plt.axis("off")
        
        # Low-Dose görüntü
        low_img = cv2.imread(os.path.join(low_dose_dir, low_dose_files[i]), cv2.IMREAD_GRAYSCALE)
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(low_img, cmap="gray")
        plt.title(f"Low-Dose: {low_dose_files[i]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Test et
if __name__ == "__main__":
    visualize_samples()
