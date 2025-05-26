import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets.pet_dataset import PETDataset
from models.generator import UNetGenerator as Generator
from models.discriminator import PatchGANDiscriminator as Discriminator

# Dataset path
low_dir = "genAI/data/low_dose"
high_dir = "genAI/data/high_dose"

dataset = PETDataset(low_dir, high_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Modeller
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Losslar
l1_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()

# Optimizerlar
optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)

# Eğitim
epochs = 10
for epoch in range(epochs):
    for batch in dataloader:
        real = batch['high'].to(device)
        low = batch['low'].to(device)

        ######################
        # Train Discriminator
        ######################
        fake = generator(low)
        pred_real = discriminator(torch.cat((low, real), dim=1))
        pred_fake = discriminator(torch.cat((low, fake.detach()), dim=1))

        loss_d = 0.5 * (bce_loss(pred_real, torch.ones_like(pred_real)) +
                        bce_loss(pred_fake, torch.zeros_like(pred_fake)))

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        ##################
        # Train Generator
        ##################
        pred_fake = discriminator(torch.cat((low, fake), dim=1))
        loss_g = bce_loss(pred_fake, torch.ones_like(pred_fake)) + l1_loss(fake, real)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss G: {loss_g.item():.4f} | Loss D: {loss_d.item():.4f}")

# Kaydet
torch.save(generator.state_dict(), "generator_gan.pth")
print("✅ Model saved.")
