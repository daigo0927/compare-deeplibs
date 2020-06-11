import cv2
import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset


def prepare_samples(dataset_dir, train_or_test, num_classes=10):
    image_paths, labels = [], []
    
    sub_dir = dataset_dir + '/' + train_or_test
    for c in range(num_classes):
        query = '/'.join([sub_dir, str(c), '*.png'])
        paths = glob(query)
        image_paths += paths
        labels += [c]*len(paths)

    return image_paths, labels


class Cifar10(Dataset):
    def __init__(self,
                 input_image_paths,
                 target_labels,
                 preprocess=None,
                 augmentation=None):
        self.input_image_paths = input_image_paths
        self.target_labels = target_labels
        self.preprocess = preprocess
        self.augmentation = augmentation

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.input_image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.preprocess:
            image = self.preprocess(image=image)['image']
                
        if self.augmentation:
            image = self.augmentation(image=image)['image']

        image = np.transpose(image, (2, 0, 1))
        label = self.target_labels[idx]

        image = torch.from_numpy(image)
        label = torch.as_tensor(label, dtype=torch.long)
        return image, label
