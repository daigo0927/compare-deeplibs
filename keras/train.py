import yaml
import argparse
import tensorflow as tf
import albumentations as A

from dataset import prepare_samples, Cifar10
from model import get_resnet_mini


def train(dataset_dir, batch_size, learning_rate, epochs):

    input_paths, target_labels = prepare_samples(dataset_dir, train_or_test='train')
    preprocess = A.Compose([A.Normalize()])
    augmentation = None
    dataset = Cifar10(input_paths, target_labels, batch_size, shuffle=True,
                      preprocess=preprocess, augmentation=augmentation)

    model = get_resnet_mini(input_shape=(32, 32, 3),
                            num_classes=10)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.sparse_categorical_accuracy]
    model.compile(optimizer, loss, metrics)

    model.fit(dataset, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', default='../cifar-10')
    parser.add_argument('-b', '--batch-size', default=32)
    parser.add_argument('-lr', '--learning-rate', default=0.001)
    parser.add_argument('-e', '--epochs', default=10)
    args = parser.parse_args()
    
    train(**vars(args))
