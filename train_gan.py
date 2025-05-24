import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from datasets.pet_dataset import PETDataset
from utils import create_synthetic_pet_images

import os
from tqdm import tqdm

# 1. Veriyi üret
create_synthetic_pet_images("data/high_dose", "data/low_dose", num_images=50)

# 2. Dataset ve Transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

dataset = PETDataset("data/low_dose", "data/high_dose", transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. Cihaz ve modeller
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

# 4. Kayıp fonksiyonları
l1_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()

# 5. Optimizerlar
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# 6. Eğitim döngüsü
epochs = 50
for epoch in range(epochs):
    gen_loss_total = 0.0
    disc_loss_total = 0.0

    for low_dose, high_dose in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        low_dose = low_dose.to(device)
        high_dose = high_dose.to(device)

        # === Train Discriminator ===
        with torch.no_grad():
            fake = generator(low_dose)
        pred_real = discriminator(high_dose)
        pred_fake = discriminator(fake.detach())

        real_labels = torch.ones_like(pred_real).to(device)
        fake_labels = torch.zeros_like(pred_fake).to(device)

        loss_real = bce_loss(pred_real, real_labels)
        loss_fake = bce_loss(pred_fake, fake_labels)
        disc_loss = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        disc_loss.backward()
        optimizer_D.step()

        # === Train Generator ===
        fake = generator(low_dose)
        pred_fake = discriminator(fake)

        adv_loss = bce_loss(pred_fake, real_labels)  # Generator wants discriminator to output 1
        l1 = l1_loss(fake, high_dose)
        gen_loss = l1 + 0.001 * adv_loss  # L1 dominant, adversarial ek katkı

        optimizer_G.zero_grad()
        gen_loss.backward()
        optimizer_G.step()

        gen_loss_total += gen_loss.item()
        disc_loss_total += disc_loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Gen Loss: {gen_loss_total:.4f} | Disc Loss: {disc_loss_total:.4f}")

# 7. Modelleri kaydet
torch.save(generator.state_dict(), "generator_gan.pth")
torch.save(discriminator.state_dict(), "discriminator_gan.pth")
print("✅ GAN modelleri kaydedildi.")
