import os, sys
sys.path.append(os.pardir)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_keras import build_resnet
from utils import prepare_parser


def train(args):
    datagen = ImageDataGenerator(
        horizontal_flip=args.flip_left_right,
        vertical_flip=args.flip_up_down,
        rescale=1/255.0,
        validation_split=args.validation_split
    )

    train_generator = datagen.flow_from_directory(
        directory=args.dataset_dir+'/train',
        target_size=args.resize_shape,
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True,
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        directory=args.dataset_dir+'/train',
        target_size=args.resize_shape,
        class_mode='categorical',
        batch_size=args.batch_size,
        subset='validation'
    )

    model = build_resnet(input_shape=train_generator.image_shape,
                         filters=args.filters,
                         output_dim=train_generator.num_classes,
                         output_type='prob')

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = tf.keras.losses.categorical_crossentropy
    accuracy = tf.keras.metrics.categorical_accuracy
    model.compile(optimizer, loss, [accuracy])

    callbacks = None
    model.fit_generator(train_generator,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=val_generator)
    # .fit_generator raises NoneType error when the iteration finished
    # in keras of tf-v1 API


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
