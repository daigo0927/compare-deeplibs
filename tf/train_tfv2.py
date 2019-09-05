import os, sys
sys.path.append(os.pardir)
import time
import numpy as np
import tensorflow as tf
from datasetv2 import Cifar10
from modelv2 import ResNetMini
from utils import show_progress, prepare_parser


loss_fn = tf.losses.categorical_crossentropy
acc_fn = tf.metrics.categorical_accuracy


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

    model = ResNetMini(filters=args.filters,
                       output_dim=dataset.num_classes,
                       name='resnet')

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    # Define parameter update operation as a static graph
    # This can speed up the execution
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = tf.reduce_mean(loss_fn(labels, logits, from_logits=True))
            acc = tf.reduce_mean(acc_fn(labels, logits))
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss, acc

    # ---------------- Actual training loop -------------------
    n_batches = np.ceil(len(dataset)*(1-args.validation_split)/args.batch_size)
    for e in range(args.epochs):
        for i, (images, labels) in enumerate(dataset.train_loader):
            start = time.time()
            loss, acc = train_step(images, labels)
            step_time = time.time() - start
            # ----- Output log -----
            if i%10 == 0:
                show_progress(e+1, i+1, int(n_batches),
                              loss=loss.numpy(), accuracy=acc.numpy(),
                              step_time=step_time)

        # -------------- Validation ---------------
        losses, accs = [], []
        for images, labels in dataset.val_loader:
            preds = model(images, training=False)
            loss = tf.reduce_mean(loss_fn(labels, preds, from_logits=True))
            acc = tf.reduce_mean(acc_fn(labels, preds))
            losses.append(loss.numpy())
            accs.append(acc.numpy())
        print('\nValidation score: loss: {}, accuracy: {}'\
              .format(np.mean(losses), np.mean(accs)))


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
