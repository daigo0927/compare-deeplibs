import numpy as np
from chainer import dataset
from PIL import Image
from glob import glob
from abc import abstractmethod, ABCMeta


class Base(dataset.DatasetMixin, metaclass=ABCMeta):
    """
    Abstract class to flexibly implement dataset pipeline
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 resize_shape=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - resize_shape: tuple for resize shape (optional)
         """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.training = True

        self.resize_shape = resize_shape
        self.transform = None

        print('Building a dataset pipeline ...')
        self._set_classes()
        self._get_filenames()
        print('Found {} images.'.format(len(self)))
        self._build_preprocess()
        print('Done.')

    def __len__(self):
        return len(self.samples)

    def get_example(self, idx):
        imagefile, label = self.samples[idx]
        image, label = self._read(imagefile, label)
        image = self.preprocess(image)
        if self.training and self.transform:
            image = self.transform(image)
        return image, label

    def _read(self, imagefile, label):
        image = Image.open(imagefile)
        label = int(label)
        return image, label

    def _build_preprocess(self):
        """ Implement preprocess (should be execute both train/val/test set) """
        preps = []
        if self.resize_shape is not None:
            preps.append(transforms.Resize(self.resize_shape))
        preps += [
            lambda x: np.asarray(x),
            lambda x: x/255.0,
            transforms.ToTensor()
        ]
        self.preprocess = transforms.Compose(preps)

    def set_transform(self, transform):
        """ Inplement input transformation """
        self.transform = transform

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


class Cifar10(Base):
    """ Chainer dataset pipeline for cifar-10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 resize_shape=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
           - resize_shape: tuple for resize shape (optional)
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         resize_shape=resize_shape)

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
