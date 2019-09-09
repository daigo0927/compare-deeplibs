import tensorflow as tf
import random
from glob import glob
from abc import abstractmethod, ABCMeta
from functools import partial


class Base(metaclass=ABCMeta):
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 one_hot=True,
                 validation_split=0.1,
                 resize_shape=None,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - one_hot: boolean for one-hot encoding
          - validation_split: validation split ratio, should be in [0, 1]
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.validation_split = validation_split

        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.rotate = rotate
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

        print('Building a dataset pipeline ...')
        self._set_classes()
        self._get_filenames()
        print('Found {} images.'.format(len(self)))
        self._build()
        print('Done.')

    def __len__(self):
        return len(self.samples[0])

    @abstractmethod
    def _set_classes(self):
        """ implement self.classes """; ...

    @abstractmethod
    def _get_filenames(self):
        """ implement self.samples """; ...
        
    @property
    def num_classes(self):
        return len(self.classes)

    def read(self, imagefile, label):
        image = tf.image.decode_image(tf.io.read_file(imagefile))
        image = tf.cast(image, tf.float32)
        return image, label

    def preprocess(self, image, label):
        if self.resize_shape is not None:
            th, tw = self.resize_shape
            image = tf.image.resize_image_with_pad(image, th, tw)
            image.set_shape((th, tw, 3))
        if self.one_hot:
            label = tf.one_hot(label, self.num_classes)
        image /= 255.0
        return image, label

    def transform(self, image, label):
        if self.crop_shape is not None:
            image = tf.image.random_crop(image, (*self.crop_shape, 3))

        if self.rotate:
            rotate_fn = tf.keras.preprocessing.image.random_rotation
            rotate_fn = partial(rotate_fn, rg=45,
                                row_axis=0, col_axis=1, channel_axis=2)
            image = tf.py_func(rotate_fn, [image], image.dtype)

        if self.flip_left_right:
            image = tf.image.random_flip_left_right(image)
        if self.flip_up_down:
            image = tf.image.random_flip_up_down(image)
            
        return image, label

    def _build(self):
        if self.train_or_test == 'train':
            idx = int(len(self) * (1 - self.validation_split))
            dataset = tf.data.Dataset.from_tensor_slices(self.samples)
            dataset = dataset.shuffle(len(self))
            self.train_iterator = dataset.take(idx)\
              .map(self.read)\
              .map(self.preprocess)\
              .map(self.transform)\
              .prefetch(self.batch_size)\
              .batch(self.batch_size)\
              .make_initializable_iterator()
            self.val_iterator = dataset.skip(idx)\
              .map(self.read)\
              .map(self.preprocess)\
              .prefetch(self.batch_size)\
              .batch(self.batch_size)\
              .make_initializable_iterator()
        else:
            self.test_iterator = tf.data.Dataset.from_tensor_slices(self.samples)\
              .map(self.read)\
              .map(self.preprocess)\
              .prefetch(self.batch_size)\
              .batch(self.batch_size)\
              .make_one_shot_iterator()
        

class DogsVsCats(Base):
    """ tf.data pipeline for dogs vs cats dataset
    https://www.kaggle.com/c/dogs-vs-cats/data
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 one_hot=True,
                 validation_split=0.1,
                 resize_shape=None,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - one_hot: boolean for one-hot encoding
          - validation_split: validation split ratio, should be in [0, 1]
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         batch_size=batch_size,
                         one_hot=one_hot,
                         validation_split=validation_split,
                         resize_shape=resize_shape,
                         crop_shape=crop_shape,
                         rotate=rotate,
                         flip_left_right=flip_left_right,
                         flip_up_down=flip_up_down)

    def _get_filenames(self):
        if self.train_or_test == 'train':
            d = self.dataset_dir + '/train'
            filepath_dog = d + '/dog*.jpg'
            filepath_cat = d + '/cat*.jpg'
            imagefiles_dog = glob(filepath_dog)
            imagefiles_cat = glob(filepath_cat)
            imagefiles =  imagefiles_dog + imagefiles_cat
            labels =  [0]*len(imagefiles_dog) + [1]*len(imagefiles_cat)
            self.samples = (imagefiles, labels)
        else:
            d = self.dataset_dir + '/test1'
            imagefiles = glob(d + '/*.jpg')
            self.samples = (imagefiles, [0]*len(imagefiles))

    def _set_classes(self):
        self.classes = ['dog', 'cat']


class Flowers(Base):
    """ tf.data pipeline for Flower dataset
    https://www.kaggle.com/alxmamaev/flowers-recognition
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 one_hot=True,
                 validation_split=0.1,
                 resize_shape=None,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - one_hot: boolean for one-hot encoding
          - validation_split: validation split ratio, should be in [0, 1]
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         batch_size=batch_size,
                         one_hot=one_hot,
                         validation_split=validation_split,
                         resize_shape=resize_shape,
                         crop_shape=crop_shape,
                         rotate=rotate,
                         flip_left_right=flip_left_right,
                         flip_up_down=flip_up_down)

    def _set_classes(self):
        self.classes = ['daisy',
                        'dandelion',
                        'rose',
                        'sunflower',
                        'tulip']

    def _get_filenames(self):
        if self.train_or_test == 'train':
            imagefiles, labels = [], []
            for l, c in enumerate(self.classes):
                filepath = '/'.join([self.dataset_dir, c, '*.jpg'])
                ifiles = glob(filepath)
                imagefiles += ifiles
                labels += [l]*len(ifiles)
            self.samples = (imagefiles, labels)
        else:
            raise ValueError('Flower dataset does not have test-set')


class Cifar10(Base):
    """ tf.data pipeline for cifar-10 dataset
    https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 one_hot=True,
                 validation_split=0.1,
                 resize_shape=None,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - one_hot: boolean for one-hot encoding
          - validation_split: validation split ratio, should be in [0, 1]
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         batch_size=batch_size,
                         one_hot=one_hot,
                         validation_split=validation_split,
                         resize_shape=resize_shape,
                         crop_shape=crop_shape,
                         rotate=rotate,
                         flip_left_right=flip_left_right,
                         flip_up_down=flip_up_down)

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
        self.samples = (imagefiles, labels)

