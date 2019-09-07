import tensorflow as tf
from tensorflow.keras import layers


def dsample(x):
    return tf.nn.avg_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')


class ResBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 downsample=False,
                 name='block'):
        super(ResBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.downsample = downsample

        self.conv0 = layers.Conv2D(filters, kernel_size, strides, 'same',
                                   name='conv0')
        self.conv1 = layers.Conv2D(filters, kernel_size, strides, 'same',
                                   name='conv1')
        self.bn0 = layers.BatchNormalization(name='bn0')
        self.bn1 = layers.BatchNormalization(name='bn1')
        if downsample:
            self.conv2 = layers.Conv2D(filters, (1, 1), name='conv2')

    def call(self, x, training):
        fx = tf.nn.relu(self.bn0(x, training=training))
        fx = self.conv0(fx)
        fx = tf.nn.relu(self.bn1(fx, training=training))
        fx = self.conv1(fx)
        if self.downsample:
            fx = dsample(fx)
            x = self.conv2(x)
            x = dsample(x)
        return x + fx


class ResNetMini(tf.keras.Model):
    def __init__(self,
                 filters,
                 output_dim,
                 output_type='logit',
                 name='resnet'):
        super(ResNetMini, self).__init__(name=name)
        self.filters = filters
        self.output_dim = output_dim
        if output_type not in ['logit', 'prob']:
            raise AssertionError('output_type is either logit or prob')
        self.output_type = output_type

        self.conv = layers.Conv2D(filters, (7, 7), (2, 2), 'same', name='conv')
        self.bn = layers.BatchNormalization(name='bn')
        self.pool = layers.MaxPooling2D((3, 3), (2, 2), 'same', name='pool')

        self.block0_0 = ResBlock(filters, name='block0_0')
        self.block0_1 = ResBlock(filters, name='block0_1')
        self.block1_0 = ResBlock(filters*2, downsample=True, name='block1_0')
        self.block1_1 = ResBlock(filters*2, name='block1_1')

        self.fc = layers.Dense(output_dim, name='fc')

    def call(self, x, training):
        h = self.conv(x)
        h = self.bn(h, training=training)
        h = self.pool(h)

        h = self.block0_0(h, training=training)
        h = self.block0_1(h, training=training)
        h = self.block1_0(h, training=training)
        h = self.block1_1(h, training=training)

        h = tf.reduce_mean(h, axis=(1, 2))
        logits = self.fc(h)
        if self.output_type == 'logit':
            return logits
        else:
            return tf.nn.softmax(logits, axis=-1)
