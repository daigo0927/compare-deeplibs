import cv2
import numpy as np
import tensorflow as tf
from glob import glob


def prepare_samples(dataset_dir, train_or_test, num_classes=10):
    image_paths, labels = [], []
    
    sub_dir = dataset_dir + '/' + train_or_test
    for c in range(num_classes):
        query = '/'.join([sub_dir, str(c), '*.png'])
        paths = glob(query)
        image_paths += paths
        labels += [c]*len(paths)

    return image_paths, labels


class Cifar10(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays).
    References: https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    def __init__(self,
                 input_image_paths,
                 target_labels,
                 batch_size=1,
                 shuffle=False,
                 preprocess=None,
                 augmentation=None):
        self.input_image_paths = input_image_paths
        self.target_labels = target_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.augmentation = augmentation

        self.on_epoch_end()
        
    def __len__(self):
        return len(self.input_image_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        x, y = [], []
        for i in self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]:
            
            image = cv2.imread(self.input_image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.preprocess:
                image = self.preprocess(image=image)['image']
                
            if self.augmentation:
                image = self.augmentation(image=image)['image']
            
            label = self.target_labels[i]

            x.append(image)
            y.append(label)

        x = np.stack(x)
        y = np.stack(y)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(range(len(self.input_image_paths)))
        else:
            self.indexes = np.arange(len(self.input_image_paths))
