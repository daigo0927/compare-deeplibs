# conding:utf-8

import os, sys
import time
import argparse
from utils import load_cifar10, load_cifar100

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, Variable
from chainer import Link, Chain
from chainer import optimizers
from chainer.cuda import to_gpu, to_cpu

class CNN(Chain):
    def __init__(self, num_output = 10):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels = 3, out_channels = 64, ksize = 5, stride = 1, pad = 2)
            self.bn1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(
                64, 128, 5, 1, 2)
            self.bn2 = L.BatchNormalization(128)
            self.fc = L.Linear(None, num_output)

    def __call__(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pooling_2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pooling_2d(x, 2, 2)
        return self.fc(x)

class Trainer(object):

    def __init__(self):

        self.net = CNN()
        self.opt = optimizers.Adam()
        self.opt.setup(self.net)
        self._load_cifar10()

    def _load_cifar10(self):
        (x_train, y_train), (x_test, y_test) = load_cifar10(to_categoric = False)
        self.x_train = np.transpose(x_train.astype('f'), (0, 3, 1, 2))
        self.y_train = y_train.flatten().astype('i')
        self.x_test = np.transpose(x_test.astype('f'), (0, 3, 1, 2))
        self.y_test = y_test.flatten().astype('i')

    def _load_cifar100(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) =\
                                        load_cifar100(to_categoric = False)
        self.x_train = np.transpose(x_train.astype('f'), (0, 3, 1, 2))
        self.y_train = y_train.flatten().astype('i')
        self.x_test = np.transpose(x_test.astype('f'), (0, 3, 1, 2))
        self.y_test = y_test.flatten().astype('i')

    def train(self,
              num_epochs,
              batch_size,
              gpu_id):

        if gpu_id is not None:
            self.net.to_gpu(gpu_id)
            self.x_train = to_gpu(self.x_train, gpu_id)
            self.y_train = to_gpu(self.y_train, gpu_id)
            self.x_test = to_gpu(self.x_test, gpu_id)
            self.y_test = to_gpu(self.y_test, gpu_id)

        num_batches = int(len(self.x_train)/batch_size)
        print('epochs : {}, number of batches : {}'\
              .format(num_epochs, num_batches))

        lap_times = []
        for e in range(num_epochs):
            permute_idx = np.random.permutation(np.arange(50000))
            lap_time = []
            for b in range(num_batches):

                x_batch = self.x_train[permute_idx[b*batch_size:(b+1)*batch_size]]
                y_batch = self.y_train[permute_idx[b*batch_size:(b+1)*batch_size]]

                s_time = time.time()
                logits = self.net(x_batch)
                loss = F.softmax_cross_entropy(logits, y_batch)
                self.net.cleargrads()
                loss.backward()
                self.opt.update()
                e_time = time.time()
                lap_time.append(e_time - s_time)

                if b%10 == 0:
                    acc = F.accuracy(logits, y_batch)
                    acc = to_cpu(acc.data)
                    print('epoch : {}, batch : {}, accuracy : {}'\
                          .format(e, b, acc))

            lap_times.append(np.sum(lap_time))

            # validation
            accs_val = []
            for b in range(int(len(self.x_test)/batch_size)):
                x_val = self.x_test[b*batch_size:(b+1)*batch_size]
                y_val = self.y_test[b*batch_size:(b+1)*batch_size]
                preds_val = self.net(x_val)
                acc_val = F.accuracy(preds_val, y_val)
                accs_val.append(to_cpu(acc_val.data))
            print('{} epoch validation accuracy {}'.format(e, np.mean(accs_val)))

            # save trained model
            serializers.save_npz('model_chainer.model', self.net)

        with open('./lap_record.csv', 'a') as f:
            f.write('chainer')
            for lap in lap_times:
                f.write(',' + str(lap))
            f.write('\n')

def train_chainer(epochs, batch_size, gpu_id):

    if not os.path.exists('./model_keras'):
        os.mkdir('./model_keras')

    trainer = Trainer()
    trainer.train(num_epochs = epochs,
                  batch_size = batch_size,
                  gpu_id = gpu_id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'number of epochs [20]')
    parser.add_argument('-b', '--batch_size', type = int, default = 64,
                        help = 'size of mini-batch [64]')
    parser.add_argument('-g', '--gpu_id', type = int, default = None,
                        help = 'utilize gpu [None]')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{} : {}'.format(key, value))

    train_chainer(**vars(args))
