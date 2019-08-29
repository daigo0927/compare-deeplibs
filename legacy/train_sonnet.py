# coding:utf-8

import numpy as np
import argparse
import time

import tensorflow as tf
import sonnet as snt

import os, sys
from utils import load_cifar10, load_cifar100, show_progress

class CNN(snt.AbstractModule):
    def __init__(self, num_output, name = 'cnn'):
        super(CNN, self).__init__(name = name)
        self._num_output = num_output

    def _build(self, inputs):
        x = snt.Conv2D(output_channels = 64,
                       kernel_shape = (5, 5),
                       stride = (1, 1),
                       padding = 'SAME')(inputs)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(x)
        x = snt.Conv2D(output_channels = 128,
                       kernel_shape = (5, 5),
                       stride = (1, 1),
                       padding = 'SAME')(x)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = tf.layers.MaxPooling2D((2, 2), (2, 2), 'same')(x)
        x = snt.BatchFlatten()(x)
        logits = snt.Linear(output_size = self._num_output)(x)

        return logits

class Trainer(object):
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.sess = tf.Session()
        self._load_cifar10()
        self._build_graph()

    def _load_cifar10(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()

    def _load_cifar100(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar100()

    def _build_graph(self):

        self.images = tf.placeholder(tf.float32, shape = (self.batch_size, 32, 32, 3),
                                     name = 'images')
        self.labels = tf.placeholder(tf.float32, shape = (self.batch_size, 10),
                                     name = 'labels')

        self.net = CNN(num_output = 10)
        self.logits = self.net(self.images)

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)

        self.preds = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.reduce_sum(self.labels*self.preds, axis = 1))
        
        self.opt = tf.train.AdamOptimizer()\
                           .minimize(self.loss, var_list = self.net.get_variables())

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        num_batches = int(len(self.x_train)/self.batch_size)
        print('epochs : {}, number of batches : {}'\
              .format(self.epochs, num_batches))

        lap_times = []
        # training iteration
        for e in range(self.epochs):
            permute_idx = np.random.permutation(np.arange(50000))
            lap_time = []
            for b in range(num_batches):

                x_batch = self.x_train[permute_idx[b*self.batch_size:(b+1)*self.batch_size]]
                y_batch = self.y_train[permute_idx[b*self.batch_size:(b+1)*self.batch_size]]

                s_time = time.time()
                _, loss, acc = self.sess.run([self.opt, self.loss, self.accuracy],
                                             feed_dict = {self.images:x_batch, self.labels:y_batch})
                e_time = time.time()
                lap_time.append(e_time - s_time)

                if b%10 == 0:
                    show_progress(e+1, b+1, num_batches, loss, acc)

            # record single epoch training lap-time
            lap_times.append(np.sum(lap_time))
            
            # validation
            accs_val = []
            for b in range(int(len(self.x_test)/self.batch_size)):
                x_val = self.x_test[b*self.batch_size:(b+1)*self.batch_size]
                y_val = self.y_test[b*self.batch_size:(b+1)*self.batch_size]
                acc_val = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_val,
                                                     self.labels:y_val})
                accs_val.append(acc_val)
            print('\n{} epoch validation accuracy {}'.format(e+1, np.mean(accs_val)))

            # save trained model
            self.saver.save(self.sess, './model_snt/model{}.ckpt'.format(e))

        # record training time
        with open('./lap_record.csv', 'a') as f:
            f.write('sonnet')
            for lap in lap_times:
                f.write(',' + str(lap))
            f.write('\n')

def train_tf(epochs, batch_size, gpu_id):

    if not os.path.exists('./model_snt'):
        os.mkdir('./model_snt')

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    trainer = Trainer(epochs, batch_size)
    trainer.train()
                  
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # optimization
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'number of epochs [20]')
    parser.add_argument('-b', '--batch_size', type = int, default = 128,
                        help = 'size of mini-batch [128]')
    parser.add_argument('-g', '--gpu_id', type = str, required = True,
                        help = 'utilize gpu number')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{} : {}'.format(key, value))

    train_tf(**vars(args))
