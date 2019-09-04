import os, sys
sys.path.append(os.pardir)
import utils
import tensorflow as tf
from datasetv2 import Cifar10
from model_keras import ResNetMini


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

    model = ResNetMini(output_dim=dataset.num_classes,
                       output_type='prob',
                       name='resnet')

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = tf.keras.losses.categorical_crossentropy
    accuracy = tf.keras.metrics.categorical_accuracy
    model.compile(optimizer, loss, [accuracy])

    callbacks = None
    model.fit_generator(dataset.train_loader,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=dataset.val_loader)


if __name__ == '__main__':
    parser = utils.prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
