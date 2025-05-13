import numpy as np
import cv2
import os
import glob

def create_synthetic_pet_images(high_dose_dir, low_dose_dir, num_images=50, dose_reduction_factor=0.05):
    os.makedirs(high_dose_dir, exist_ok=True)
    os.makedirs(low_dose_dir, exist_ok=True)

    for i in range(num_images):
        # 1. High-Dose Görüntü Üret (Daha Gerçekçi)
        base_image = np.zeros((128, 128), dtype=np.float32)
        cv2.circle(base_image, (64, 64), 40, (1.0), -1)  # Ana tümör bölgesi
        cv2.circle(base_image, (96, 96), 20, (0.7), -1)  # Küçük metastaz
        cv2.circle(base_image, (32, 96), 15, (0.5), -1)  # Başka küçük metastaz
        high_dose = (base_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(high_dose_dir, f"highdose_{i:03d}.png"), high_dose)

        # 2. Low-Dose Simülasyonu
        low_dose = simulate_low_dose_pet(high_dose, dose_reduction_factor)
        cv2.imwrite(os.path.join(low_dose_dir, f"highdose_{i:03d}.png"), low_dose)

    print("✅ Daha gerçekçi High-Dose ve Low-Dose görüntüler üretildi.")

def simulate_low_dose_pet(high_dose_img, dose_reduction_factor=0.05):
    """
    Daha gerçekçi low-dose PET simülasyonu:
    - Doz azaltma faktörü
    - Poisson noise (foton sayımı)
    - Anatomik yapıları koruyan bulanıklaştırma
    """
    # Float32'ye dönüştür ve normalize et
    img = high_dose_img.astype(np.float32) / 255.0
    
    # Doz azaltma
    scaled = img * dose_reduction_factor
    
    # Poisson noise ekle (foton sayımı benzetmesi)
    noisy = np.random.poisson(scaled * 1000) / 1000.0
    
    # Anatomik yapıları koruyan bulanıklaştırma
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0.5)
    
    # 0-255 aralığına geri döndür
    return np.clip(blurred * 255, 0, 255).astype(np.uint8)

# Test etmek istersen:
if __name__ == "__main__":
    create_synthetic_pet_images("data/high_dose", "data/low_dose", num_images=5)
