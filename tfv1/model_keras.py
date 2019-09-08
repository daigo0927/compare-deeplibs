import tensorflow as tf
from tensorflow.keras import Model, Input, layers


def resblock(x,
             filters,
             kernel_size=(3, 3),
             strides=(1, 1),
             downsample=False):
    fx = layers.BatchNormalization()(x)
    fx = layers.ReLU()(fx)
    fx = layers.Conv2D(filters, kernel_size, strides, 'same')(fx)
    fx = layers.BatchNormalization()(fx)
    fx = layers.ReLU()(fx)
    fx = layers.Conv2D(filters, kernel_size, strides, 'same')(fx)
    if downsample:
        fx = layers.AvgPool2D((2, 2), (2, 2))(fx)
        x = layers.Conv2D(filters, kernel_size, strides, 'same')(x)
        x = layers.AvgPool2D((2, 2), (2, 2))(x)
    return layers.Add()([x, fx])


def build_resnet(input_shape,
                 filters,
                 output_dim,
                 output_type='logit'):
    x = Input(shape=input_shape)
    
    h = layers.Conv2D(filters, (7, 7), (2, 2), 'same')(x)
    h = layers.BatchNormalization()(h)
    h = layers.MaxPooling2D((3, 3), (2, 2), 'same')(h)

    h = resblock(h, filters)
    h = resblock(h, filters)
    h = resblock(h, filters*2, downsample=True)
    h = resblock(h, filters*2)

    h = layers.GlobalAveragePooling2D()(h)
    logits = layers.Dense(output_dim)(h)
    if output_type == 'logit':
        outputs = logits
    elif output_type == 'prob':
        outputs = layers.Softmax(axis=-1)(logits)
    else:
        raise KeyError('output_tyep should be either logit or prob')
    return Model(x, outputs)
