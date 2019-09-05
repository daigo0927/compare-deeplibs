import os, sys
sys.path.append(os.pardir)
import time
import numpy as np
import tensorflow as tf
from dataset_tf import Cifar10
from model_tf import ResNetMini
from utils import show_progress, prepare_parser


loss_fn = tf.losses.softmax_cross_entropy

def acc_fn(labels, logits):
    corrects = tf.equal(tf.argmax(labels, -1), tf.argmax(logits, -1))
    accuracy = tf.reduce_mean(tf.cast(corrects, dtype=tf.float32))
    return accuracy


def train(args):
    dataset = Cifar10(dataset_dir=args.dataset_dir,
                      train_or_test='train',
                      batch_size=args.batch_size,
                      one_hot=True,
                      validation_split=args.validation_split,
                      resize_shape=args.resize_shape,
                      crop_shape=args.crop_shape,
                      rotate=args.rotate,
                      flip_left_right=args.flip_left_right,
                      flip_up_down=args.flip_up_down)

    # ---------- Static Graph definition -----------
    sess = tf.Session()

    model = ResNetMini(filters=args.filters,
                       output_dim=dataset.num_classes,
                       name='resnet')
    # Training graph
    images, labels = dataset.train_iterator.get_next()
    logits = model(images, training=True)
    loss = loss_fn(labels, logits)
    acc = acc_fn(labels, logits)

    # Validation graph
    images_val, labels_val = dataset.val_iterator.get_next()
    logits_val = model(images_val, training=False)
    loss_val = loss_fn(labels_val, logits_val)
    acc_val = acc_fn(labels_val, logits_val)

    # Set parameter update operations
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    update_weights = optimizer.minimize(loss, var_list=model.trainable_weights)
    update_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    saver = tf.train.Saver(model.weights)
    # Initializer variables
    sess.run(tf.global_variables_initializer())

    # ---------------- Actual training loop -------------------
    n_batches = int(len(dataset)*(1-args.validation_split)/args.batch_size)
    n_batches_val = int(len(dataset)*args.validation_split/args.batch_size)
    for e in range(args.epochs):
        for i in range(n_batches):
            start = time.time()
            sess.run([update_weights, update_bn])
            step_time = time.time() - start
            # ----- Output log -----
            if i%10 == 0:
                loss_, acc_ = sess.run([loss, acc])
                show_progress(e+1, i+1, int(n_batches),
                              loss=loss_, accuracy=acc_,
                              step_time=step_time)

        # -------------- Validation ---------------
        losses, accs = [], []
        for _ in range(n_batches_val):
            loss_, acc_ = sess.run([loss_val, acc_val])
            losses.append(loss_)
            accs.append(acc_)
        print('\nValidation score: loss: {}, accuracy: {}'\
              .format(np.mean(losses), np.mean(accs)))


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
