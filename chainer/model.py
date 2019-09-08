import chainer
import chainer.functions as F
import chainer.links as L
from functools import partial


class ResBlock(chainer.Chain):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.downsample = downsample

        with self.init_scope():
            self.conv_0 = L.Convolution2D(in_channels, out_channels,
                                          ksize, stride, pad=1)
            self.conv_1 = L.Convolution2D(out_channels, out_channels,
                                          ksize, stride, pad=1)
            self.bn_0 = L.BatchNormalization(in_channels)
            self.bn_1 = L.BatchNormalization(out_channels)
            if downsample:
                self.conv_2 = L.Convolution2D(in_channels, out_channels,
                                              1, 1, 0)

    def forward(self, x):
        fx = F.relu(self.bn_0(x))
        fx = self.conv_0(fx)
        fx = F.relu(self.bn_1(fx))
        fx = self.conv_1(fx)
        if self.downsample:
            fx = F.average_pooling_2d(fx, 2, 2)
            x = F.average_pooling_2d(self.conv_2(x), 2, 2)
        return x + fx


class ResNetMini(chainer.Chain):
    def __init__(self,
                 filters,
                 output_dim):
        super().__init__()
        self.filters = filters
        self.output_dim = output_dim

        with self.init_scope():
            self.features = chainer.Sequential(
                L.Convolution2D(3, filters, 7, 2, 3),
                L.BatchNormalization(filters),
                partial(F.average_pooling_2d, ksize=3, stride=2, pad=1),
                ResBlock(filters, filters),
                ResBlock(filters, filters),
                ResBlock(filters, filters*2, downsample=True),
                ResBlock(filters*2, filters*2),
                partial(F.average, axis=(2, 3))
                )
            self.fc = F.Linear(filters*2, output_dim)

    def forward(self, x):
        x = self.features(x)
        logits = self.fc(x)
        return logits
