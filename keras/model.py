import tensorflow as tf
from tensorflow.keras import layers


class ResBlock(layers.Layer):
    def __init__(self,
                 filters,
                 strides=1):
        super(ResBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        
        self.conv0 = layers.Conv2D(filters, 3, strides, 'same')
        self.conv1 = layers.Conv2D(filters, 3, 1, 'same')
        self.bn0 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        if strides != 1:
            self.convsc = layers.Conv2D(filters, 1, strides)
            self.bnsc = layers.BatchNormalization()

    def call(self, x, training):
        fx = self.conv0(x)        
        fx = self.bn0(fx, training=training)
        fx = tf.nn.relu(fx)
        fx = self.conv1(fx)
        fx = self.bn1(fx, training=training)
        
        if self.strides != 1:
            x = self.convsc(x)
            x = self.bnsc(x, training=training)

        out = x + fx
        out = tf.nn.relu(out)
        return out


def get_resnet_mini(input_shape, num_classes):
    ''' 
    - Official ResNet-50
      http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
    - PyTorch implementation
      https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    '''
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(64, 7, 2, 'same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D((3, 3), (2, 2), 'same'),
        ResBlock(64),
        ResBlock(64),
        ResBlock(128, strides=2),
        ResBlock(128),
        layers.GlobalAvgPool2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
