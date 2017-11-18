# coding:utf-8

import numpy as np
import argparse
import time

import tensorflow as tf
import tensorflow.contrib.layers as tcl

import os, sys
sys.path.append(os.pardir)
from utils import load_cifar10, load_cifar100

class CNN(object):
    def __init__(self,
                 num_output,
                 name = 'cnn'):
        self.num_output = num_output
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(self.name) as vs:

            x = tcl.conv2d(inputs,
                           num_outputs = 64,
                           kernel_size = (5, 5),
                           stride = (1, 1),
                           padding = 'SAME')
            x = tcl.batch_norm(x)
            x = tf.nn.relu(x)
            x = tcl.max_pool2d(x, (2, 2), (2, 2), 'SAME')
            x = tcl.conv2d(x,
                           num_outputs = 128,
                           kernel_size = (5, 5),
                           stride = (1, 1),
                           padding = 'SAME')
            x = tcl.batch_norm(x)
            x = tf.nn.relu(x)
            x = tcl.max_pool2d(x, (2, 2), (2, 2), 'SAME')
            x = tcl.flatten(x)
            logits = tcl.fully_connected(x, num_outputs = self.num_output,
                                         activation_fn = None)
            return logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Trainer(object):

    def __init__(self):

        self.sess = tf.Session()
        self._load_cifar10()
        self._build_graph()

    def _load_cifar10(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()

    def _load_cifar100(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar100()

    def _build_graph(self):

        self.images = tf.placeholder(tf.float32,
                                     shape = (None, 32, 32, 3), name = 'images')
        self.labels = tf.placeholder(tf.float32,
                                     shape = (None, 10), name = 'labels')

        self.net = CNN(num_output = 10)
        self.logits = self.net(self.images)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels = self.labels, logits = self.logits))

        self.preds = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.reduce_sum(self.labels*self.preds, axis = 1))
        
        self.opt = tf.train.AdamOptimizer()\
                           .minimize(self.loss, var_list = self.net.vars)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def train(self,
              num_epochs,
              batch_size):

        num_batches = int(len(self.x_train)/batch_size)
        print('epochs : {}, number of batches : {}'\
              .format(num_epochs, num_batches))

        lap_times = []
        # training iteration
        for e in range(num_epochs):
            permute_idx = np.random.permutation(np.arange(50000))
            lap_time = []
            for b in range(num_batches):

                x_batch = self.x_train[permute_idx[b*batch_size:(b+1)*batch_size]]
                y_batch = self.y_train[permute_idx[b*batch_size:(b+1)*batch_size]]

                s_time = time.time()
                self.sess.run(self.opt,
                              feed_dict = {self.images:x_batch, self.labels:y_batch})
                e_time = time.time()
                lap_time.append(e_time - s_time)

                if b%10 == 0:
                    acc = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_batch,
                                                     self.labels:y_batch})
                    print('epoch : {}, batch : {}, accuracy : {}'\
                          .format(e, b, acc))

            # record single epoch training lap-time
            lap_times.append(np.sum(lap_time))
            
            # validation
            accs_val = []
            for b in range(int(len(self.x_test)/batch_size)):
                x_val = self.x_test[b*batch_size:(b+1)*batch_size]
                y_val = self.y_test[b*batch_size:(b+1)*batch_size]
                acc_val = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_val,
                                                     self.labels:y_val})
                accs_val.append(acc_val)
            print('{} epoch validation accuracy {}'.format(e, np.mean(accs_val)))

            # save trained model
            self.saver.save(self.sess, './model_tf/model{}.ckpt'.format(e))

        # record training time
        with open('./lap_record.csv', 'a') as f:
            f.write('tensorflow')
            for lap in lap_times:
                f.write(',' + str(lap))
            f.write('\n')

def train_tf(epochs, batch_size, gpu_id):

    if not os.path.exists('./model_tf'):
        os.mkdir('./model_tf')

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    trainer = Trainer()
    trainer.train(num_epochs = epochs,
                  batch_size = batch_size)
                  
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
