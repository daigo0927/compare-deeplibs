import numpy as np
from chainer import dataset
from PIL import Image
import albumentations as A
from glob import glob
from abc import abstractmethod, ABCMeta


class Base(dataset.DatasetMixin, metaclass=ABCMeta):
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
          - resize_shape: tuple for resize shape (optional)
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

    def get_example(self, idx):
        imagefile, label = self.samples[idx]
        image, label = self._read(imagefile, label)
        if self.preprocess:
            image = self.preprocess(image=image)['image']
        if self.training and self.transform:
            image = self.transform(image=image)['image']
        image = np.transpose(image, (2, 0, 1))
        return image, label

    def _read(self, imagefile, label):
        image = np.asarray(Image.open(imagefile))
        label = int(label)
        return image, label

    def _set_classes(self):
        """ implement self.classes """; ...

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



class Preprocess:
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape
        prep = [A.Resize(*resize_shape)] if self.resize_shape else []
        prep.append(A.Normalize((0, 0, 0), (1, 1, 1), max_pixel_value=255.0))
        self.preprocess = A.Compose(prep)

    def __call__(self, **kwargs):
        return self.preprocess(**kwargs)


class Transform:
    def __init__(self,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        self.crop_shape = crop_shape
        self.rotate = rotate
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

        trans = []
        if crop_shape: trans.append(A.RandomCrop(*crop_shape))
        if rotate: trans.append(A.Rotate(45))
        if flip_left_right: trans.append(A.HorizontalFlip())
        if flip_up_down: trans.append(A.VerticalFlip())
        self.transform = A.Compose(trans)

    def __call__(self, **kwargs):
        return self.transform(**kwargs)
     

class Cifar10(Base):
    """ Chainer dataset pipeline for cifar-10 dataset
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
