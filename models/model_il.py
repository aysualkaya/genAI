import torch
import torch.nn as nn

class ModelILGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super(ModelILGenerator, self).__init__()

        self.encoders = nn.ModuleList()
        for feature in features:
            self.encoders.append(self._block(in_channels, feature))
            in_channels = feature

        self.bottleneck = self._block(features[-1], features[-1]*2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoders.append(self._block(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.upsamples)):
            x = self.upsamples[idx](x)
            if x.shape != skip_connections[idx].shape:
                x = nn.functional.interpolate(x, size=skip_connections[idx].shape[2:])
            x = torch.cat((skip_connections[idx], x), dim=1)
            x = self.decoders[idx](x)

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
