import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample = downsample

        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding=1)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size,
                                stride, padding=1)
        self.bn_0 = nn.BatchNorm2d(in_channels)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if downsample:
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        fx = self.relu(self.bn_0(x))
        fx = self.conv_0(fx)
        fx = self.relu(self.bn_1(fx))
        fx = self.conv_1(fx)
        if self.downsample:
            fx = self.pool(fx)
            x = self.pool(self.conv_2(x))
        return x + fx


class ResNetMini(nn.Module):
    def __init__(self,
                 filters,
                 output_dim):
        super().__init__()
        self.filters = filters
        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, 7, 2, 3),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(filters, filters),
            ResBlock(filters, filters),
            ResBlock(filters, filters*2, downsample=True),
            ResBlock(filters*2, filters*2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(filters*2, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.filters*2)
        logits = self.fc(x)
        return logits
