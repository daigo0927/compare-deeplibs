import os, sys
sys.path.append(os.pardir)
import utils
import numpy as np
import tensorflow as tf
from datasetv2 import Cifar10
from modelv2 import ResNetMini


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

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            y_pred = model(images, training=True)
            loss = tf.reduce_mean(loss_fn(labels, y_pred, from_logits=True))
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    for e in range(args.epochs):
        for i, (images, labels) in enumerate(dataset.train_loader):
            loss = train_step(images, labels)

            if i%10 == 0:
                utils.show_progress(e+1, i+1, len(dataset)//args.batch_size,
                                    loss=loss.numpy())

    losses, accs = [], []
    for images, labels in dataset.val_loader:
        preds = model(images, training=False)
        loss = tf.reduce_mean(loss_fn(labels, preds))
        acc = tf.reduce_mean(acc_fn(labels, preds))
        losses.append(loss.numpy())
        accs.append(acc.numpy())
    print(f'Validation score: loss: {np.mean(losses)}, accuracy: {np.mean(accs)}')

    # model.save_weights('./modelv2.ckpt')


if __name__ == '__main__':
    parser = utils.prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
