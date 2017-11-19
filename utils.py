# coding:utf-8

import sys
import numpy as np
import math
import tensorflow as tf
from scipy.misc import imread, imresize

from keras.datasets import cifar10, cifar100
from keras.utils.np_utils import to_categorical

def leaky_relu(leak = 0.2):
    def f(inputs):
        return tf.maximum(tf.minimum(0., leak*inputs), inputs)
    return f

def tf_image_label_concat(image, label,
                          image_size, label_size):

    label_ = tf.reshape(label, (-1, 1, 1, label_size))
    label_panel = tf.tile(label_, # shape(None, width, width, feature)
                          multiples = [1, image_size, image_size, 1])
    image_label = tf.concat((image, label_panel), axis = 3)

    return image_label # shape(None, width, width, channel+feature)

def load_cifar10(to_categoric = True):
    print('load cifar10 data ...')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if to_categoric:
        y_train = to_categorical(y_train, num_classes = 10)
        y_test = to_categorical(y_test, num_classes = 10)

    return (x_train/255., y_train), (x_test/255., y_test)

def load_cifar100(to_categoric = True):
    print('load cifar100 data ...')
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    if to_categoric:
        y_train = to_categorical(y_train, num_classes = 100)
        y_test = to_categorical(y_test, num_classes = 100)

    return (x_train/255., y_train), (x_test/255., y_test)

def cifar10_extract(label = 'cat'):
    # acceptable label
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    target_label = labels.index(label)

    (x_train, t_train), (x_test, t_test) = cifar10.load_data()

    t_target = t_train==target_label
    t_target = t_target.reshape(t_target.size)

    x_target = x_train[t_target]

    print('extract {} labeled images, shape(5000, 32, 32, 3)'.format(label))
    return x_target


# shape(generated_images) : (sample_num, w, h, 3)
def combine_images(generated_images):

    total, width, height, ch = generated_images.shape
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)

    combined_image = np.zeros((height*rows, width*cols, 3),
                              dtype = generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1), :]\
            = image

    return combined_image

def get_image(filepath, image_target, image_size):

    img = imread(filepath).astype(np.float)
    h_origin, w_origin = img.shape[:2]

    if image_target > h_origin or image_target > w_origin:
        image_target = min(h_origin, w_origin)

    h_drop = int((h_origin - image_target)/2)
    w_drop = int((w_origin - image_target)/2)

    if img.ndim == 2:
        img = np.tile(img.reshape(h_origin, w_origin, 1), (1,1,3))

    img_crop = img[h_drop:h_drop+image_target, w_drop:w_drop+image_target, :]

    img_resize = imresize(img_crop, [image_size, image_size])

    return np.array(img_resize)/127.5 - 1.

def show_progress(e,b,b_total,loss,acc):
    sys.stdout.write("\r%3d: [%5d / %5d] loss: %f acc: %f" % (e,b,b_total,loss,acc))
    sys.stdout.flush()

