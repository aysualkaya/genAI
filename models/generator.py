import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetGenerator, self).__init__()
        
        # Encoder (Downsampling)
        self.encoders = nn.ModuleList()
        for feature in features:
            self.encoders.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder (Upsampling)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for feature in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoders.append(self._block(feature * 2, feature))

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder (Downsampling)
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder (Upsampling)
        for i in range(len(self.decoders)):
            x = self.upsamples[i](x)
            skip_connection = skip_connections[i]
            
            # Eğer boyutlar uyuşmuyorsa, cropping yap
            if x.size() != skip_connection.size():
                x = nn.functional.interpolate(x, size=skip_connection.size()[2:])
            
            # Skip bağlantıları ekle
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[i](x)

        # Son Convolution (Tek kanal, 1-256 aralığına dönüş)
        return torch.sigmoid(self.final_conv(x))

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

# Test etmek istersen
if __name__ == "__main__":
    model = UNetGenerator()
    x = torch.randn((1, 1, 128, 128))  # 1 kanal, 128x128 çözünürlük
    out = model(x)
    print(out.shape)  # Beklenen çıktı: (1, 1, 128, 128)
