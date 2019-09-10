import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from abc import abstractmethod, ABCMeta


class Base(Dataset, metaclass=ABCMeta):
    """
    Abstract class to flexibly implement dataset pipeline
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 preprocess=None,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - preprocess: preprocess applied to ALL data
          - transform: data augmentation applied to TRAINING data
         """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.training = True

        self.preprocess = preprocess
        self.transform = transform

        print('Building a dataset pipeline ...')
        self._set_classes()
        self._get_filenames()
        print('Found {} images.'.format(len(self)))
        print('Done.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imagefile, label = self.samples[idx]
        image, label = self._read(imagefile, label)
        label = torch.as_tensor(label, dtype=torch.long)
        if self.preprocess:
            image = self.preprocess(image)
        if self.training and self.transform:
            image = self.transform(image)
        return image, label

    def _read(self, imagefile, label):
        image = Image.open(imagefile)
        label = int(label)
        return image, label

    @abstractmethod
    def _set_classes(self):
        """ implement self.classes """; ...

    @abstractmethod
    def _get_filenames(self):
        """ implement self.samples """; ...

    @property
    def num_classes(self):
        return len(self.classes)

    def eval(self):
        """ Set to evaluation mode (skip augumentation) """
        self.training = False

    def train(self):
        """ Set to training (default) mode (executes augumentation) """
        self.training = True


class Cifar10(Base):
    """ torch dataset pipeline for cifar-10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 preprocess=None,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - preprocess: preprocess applied to ALL data
          - transform: data augmentation applied to TRAINING data
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         preprocess=preprocess,
                         transform=transform)

    def _set_classes(self):
        self.classes = ['airplane',
			'automobile',
			'bird',
			'cat',
			'deer',
			'dog',
			'frog',
			'horse',
			'ship',
			'truck']

    def _get_filenames(self):
        subd = self.dataset_dir + '/' + self.train_or_test
        imagefiles, labels = [], []
        for c in range(self.num_classes):
            filepath = '/'.join([subd, str(c), '*.png'])
            ifiles = glob(filepath)
            imagefiles += ifiles
            labels += [c]*len(ifiles)
        self.samples = [(file, label) for file, label in zip(imagefiles, labels)]
