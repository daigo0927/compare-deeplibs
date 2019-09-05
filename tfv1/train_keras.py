import os, sys
sys.path.append(os.pardir)
import tensorflow as tf
from tensorflow.keras import Input, Model
from dataset_tf import Cifar10
from model_keras import ResNetMini
from utils import prepare_parser


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

    inputs = Input(shape=(*args.resize_shape, 3))
    net = ResNetMini(filters=args.filters,
                     output_dim=dataset.num_classes,
                     output_type='prob',
                     name='resnet')
    outputs = net(inputs, training=True)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = tf.keras.losses.categorical_crossentropy
    accuracy = tf.keras.metrics.categorical_accuracy
    model.compile(optimizer, loss, [accuracy])

    callbacks = None
    n_batches = int(len(dataset)*(1-args.validation_split)/args.batch_size)
    model.fit_generator(dataset.train_iterator,
                        steps_per_epoch=n_batches,
                        epochs=args.epochs,
                        callbacks=callbacks)
                        # validation_data=val_generator)
    # validation_data raises NoneType error in keras of tf-v1 API


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
