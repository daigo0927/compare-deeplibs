# conding:utf-8

import os, sys
sys.path.append(os.pardir)
import time
import argparse

import numpy as np
import h5py

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from misc.utils import load_cifar10, load_cifar100

def cnn(image_size = 32, num_output = 10):

    w = int(image_size)

    inputs = Input(shape = (w, w, 3))
    x = Conv2D(64, (4, 4), strides = (1, 1), padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(num_output)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    
    return model

class Trainer(object):

    def __init__(self):

        self.net = cnn()
        self.opt = Adam()
        self.net.compile(loss = 'categorical_crossentropy', optimizer = self.opt)
        self._load_cifar10()

    def _load_cifar10(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()

    def _load_cifar100(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar100()

    def train(self,
              num_epochs,
              batch_size):

        num_batches = int(len(self.x_train)/batch_size)
        print('epochs : {}, number of baches : {}'\
              .format(num_epochs, num_batches))

        lap_times = []
        for e in range(num_epochs):
            permute_idx = np.random.permutation(np.arange(50000))
            lap_time = []
            for b in range(num_batches):

                x_batch = self.x_train[permute_idx[b*batch_size:(b+1)*batch_size]]
                y_batch = self.y_train[permute_idx[b*batch_size:(b+1)*batch_size]]

                s_time = time.time()
                loss = self.net.train_on_batch(x_batch, y_batch)
                e_time = time.time()
                lap_time.append(e_time - s_time)

                if b%10 == 0:
                    preds = self.net.predict(x_batch)
                    acc = np.mean(np.sum(preds*y_batch, axis = 1))
                    print('epoch : {}, batch : {}, accuracy : {}'\
                          .format(e, b, acc))

            lap_times.append(np.sum(lap_time))

            # validation
            accs_val = []
            for b in range(int(len(self.x_test)/batch_size)):
                x_val = self.x_test[b*batch_size:(b+1)*batch_size]
                y_val = self.y_test[b*batch_size:(b+1)*batch_size]
                preds_val = self.net.predict(x_val)
                acc_val = np.mean(np.sum(preds_val*y_val, axis = 1))
                accs_val.append(acc_val)
            print('{} epoch validation accuracy {}'.format(e, np.mean(accs_val)))

            # save trained model
            self.net.save_weights('./model_keras/model_{}.h5'.format(e))

        with open('./lap_record.csv', 'a') as f:
            f.write('keras')
            for lap in lap_times:
                f.write(',' + str(lap))
            f.write('\n')

def train_keras(epochs, batch_size):

    if not os.path.exists('./model_keras'):
        os.mkdir('./model_keras')

    trainer = Trainer()
    trainer.train(num_epochs = epochs,
                  batch_size = batch_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'number of epochs [20]')
    parser.add_argument('-b', '--batch_size', type = int, default = 64,
                        help = 'size of mini-batch [64]')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{} : {}'.format(key, value))

    train_keras(**vars(args))
