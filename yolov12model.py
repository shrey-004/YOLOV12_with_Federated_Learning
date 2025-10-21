import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = out_channels // 2
        self.layer1 = ConvBlock(in_channels, hidden)
        self.layer2 = ConvBlock(hidden, out_channels)

    def forward(self, x):
        return self.layer2(self.layer1(x)) + x

class YOLOv12(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        # Backbone
        self.layer1 = ConvBlock(3, 32, 3, 1)
        self.layer2 = ConvBlock(32, 64, 3, 2)
        self.layer3 = CSPBlock(64, 64)
        self.layer4 = ConvBlock(64, 128, 3, 2)
        self.layer5 = CSPBlock(128, 128)
        self.layer6 = ConvBlock(128, 256, 3, 2)
        self.layer7 = CSPBlock(256, 256)
        # Detection Head
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, (num_classes + 5) * 3, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return self.head(x)
