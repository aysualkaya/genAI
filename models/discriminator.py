import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super(PatchGANDiscriminator, self).__init__()
        
        layers = []
        for feature in features:
            layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = feature
        
        # Son katman (1 kanal, tek çıkış)
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Test etmek istersen
if __name__ == "__main__":
    model = PatchGANDiscriminator()
    x = torch.randn((1, 1, 128, 128))
    out = model(x)
    print(out.shape)  # Beklenen çıktı: (1, 1, 16, 16)
