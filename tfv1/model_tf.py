import tensorflow as tf


def dsample(x):
    return tf.layers.max_pooling2d(x, (2, 2), (2, 2), 'valid')


class ResBlock:
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 downsample=False,
                 name='block'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.downsample = downsample
        self.name = name

        self.conv0 = tf.layers.Conv2D(filters, kernel_size, strides, 'same',
                                      name='conv_0')
        self.conv1 = tf.layers.Conv2D(filters, kernel_size, strides, 'same',
                                      name='conv_1')
        self.bn0 = tf.layers.BatchNormalization(name='bn_0')
        self.bn1 = tf.layers.BatchNormalization(name='bn_1')
        if downsample:
            self.conv2 = tf.layers.Conv2D(filters, (1, 1), name='conv_2')
 
    def __call__(self, x, training):
        with tf.variable_scope(self.name):
            fx = tf.nn.relu(self.bn0(x, training=training))
            fx = self.conv0(fx)
            fx = tf.nn.relu(self.bn1(fx, training=training))
            fx = self.conv1(fx)
            if self.downsample:
                fx = dsample(fx)
                x = dsample(self.conv2(x))
            return x + fx


class ResNetMini:
    def __init__(self,
                 filters,
                 output_dim,
                 output_type='logit',
                 name='resnet'):
        self.filters = filters
        self.output_dim = output_dim
        self.output_type = output_type
        self.name = name

        self.conv = tf.layers.Conv2D(filters, (7, 7), (2, 2), 'same',
                                     name='conv')
        self.bn = tf.layers.BatchNormalization(name='bn')
        self.pool = tf.layers.MaxPooling2D((3, 3), (2, 2), 'same',
                                           name='pool')

        self.block00 = ResBlock(filters, name='block_00')
        self.block01 = ResBlock(filters, name='block_01')
        self.block10 = ResBlock(filters*2, downsample=True, name='block_10')
        self.block11 = ResBlock(filters*2, name='block_11')

        self.fc = tf.layers.Dense(output_dim, name='fc')

    def __call__(self, x, training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = self.conv(x)
            h = self.pool(self.bn(h, training=training))

            h = self.block00(h, training=training)
            h = self.block01(h, training=training)
            h = self.block10(h, training=training)
            h = self.block11(h, training=training)

            h = tf.reduce_mean(h, axis=(1, 2))
            logits = self.fc(h)
            if self.output_type == 'logit':
                return logits
            elif self.output_type == 'prob':
                return tf.nn.softmax(logits, axis=-1)
            else:
                raise KeyError('output_type should either of logit or prob')

    @property
    def weights(self):
        return [v for v in tf.global_variables() if self.name in v.name]
    
    @property
    def trainable_weights(self):
        return [v for v in tf.trainable_variables() if self.name in v.name]

                
