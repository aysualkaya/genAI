import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.simple_sr import SimpleSR
from datasets.pet_dataset import PETDataset
from genAI.utils import create_synthetic_pet_images
from genAI.plot_losses import plot_losses  # ✅ Yeni import
import os
import torch.nn as nn
import torch.optim as optim

# 1. Veri üret (GÜNCELLENDİ)
create_synthetic_pet_images("data/high_dose", "data/low_dose", num_images=100, dose_reduction_factor=0.2)


# 2. Dataset ve Transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

dataset = PETDataset("data/low_dose", "data/high_dose", transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. Model ve Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSR().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Eğitim döngüsü
epochs = 50
losses = []  # ✅ Lossları kaydetmek için

for epoch in range(epochs):
    running_loss = 0.0
    for inp, gt in train_loader:
        inp, gt = inp.to(device), gt.to(device)

        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)  # ✅ Loss kaydetme
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    # 5. Her 10 epoch'ta bir loss grafiği kaydet
    if (epoch + 1) % 10 == 0:
        plot_losses(losses, save_path=f"training_loss_epoch_{epoch+1}.png")

# 6. Son Loss grafiği
plot_losses(losses, save_path="final_training_loss.png")

# 7. Modeli kaydet
torch.save(model.state_dict(), "simple_sr.pth")
print("✅ Model kaydedildi.")
