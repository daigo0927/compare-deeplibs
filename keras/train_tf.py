import yaml
import argparse
import tensorflow as tf
import albumentations as A

from tqdm import tqdm
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
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
            metric.update_state(y, preds)
            gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss

    for e in range(epochs):
        for i, (images, labels) in enumerate(tqdm(dataset)):
            loss = train_step(images, labels)
        accuracy = metric.result().numpy()
        print(f'Epoch: {epoch}, accuracy: {accuracy}')
        metric.reset_states()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', default='../cifar-10')
    parser.add_argument('-b', '--batch-size', default=32)
    parser.add_argument('-lr', '--learning-rate', default=0.001)
    parser.add_argument('-e', '--epochs', default=10)
    args = parser.parse_args()
    
    train(**vars(args))
