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
    initialize_train = dataset.train_iterator.initializer
    images, labels = dataset.train_iterator.get_next()
    logits = model(images, training=True)
    loss = loss_fn(labels, logits)
    acc = acc_fn(labels, logits)

    # Validation graph
    initialize_val = dataset.val_iterator.initializer
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
    start_loop = time.time()
    n_batches = int(len(dataset)*(1-args.validation_split)/args.batch_size)
    n_batches_val = int(len(dataset)*args.validation_split/args.batch_size)
    for e in range(args.epochs):
        start_epoch = time.time()
        
        sess.run(initialize_train)
        for i in range(n_batches):
            start_batch = time.time()
            _, _, loss_, acc_ = sess.run([update_weights, update_bn, loss, acc])
            batch_time = time.time() - start_batch
            # ----- Output log -----
            if i%10 == 0 or i+1 == n_batches:
                show_progress(e+1, i+1, int(n_batches),
                              loss=loss_, accuracy=acc_,
                              batch_time=batch_time)

        # -------------- Validation ---------------
        losses, accs = [], []
        sess.run(initialize_val)
        for _ in range(n_batches_val):
            loss_, acc_ = sess.run([loss_val, acc_val])
            losses.append(loss_)
            accs.append(acc_)

        epoch_time = time.time() - start_batch
        print('\nValidation score: loss: {}, accuracy: {}, epoch time: {}'\
              .format(np.mean(losses), np.mean(accs), epoch_time))

    loop_time = time.time() - start_loop
    print('Total time: {}sec.'.format(loop_time))


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
