import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1:
            self.convsc = nn.Conv2d(in_channels, out_channels, 1, stride)
            self.bnsc = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fx = self.conv0(x)
        fx = self.bn0(fx)
        fx = self.relu(fx)
        fx = self.conv1(fx)
        fx = self.bn1(fx)

        if self.stride != 1:
            x = self.convsc(x)
            x = self.bnsc(x)

        out = x + fx
        out = self.relu(out)
        return out


class ResNetMini(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, 2),
            ResBlock(128, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128)
        logits = self.fc(x)
        return logits
