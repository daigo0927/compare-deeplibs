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
        self.validation_split = validation_split

        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.rotate = rotate
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

        print('Building a dataset pipeline ...')
        self._get_filenames()
        self._get_classes()
        print('Found {} images.'.format(len(self)))
        self._build()
        print('Done.')

    def __len__(self):
        return len(self.samples[0])

    @abstractmethod
    def _get_filenames(self):
        """ implement self.samples """; ...

    @abstractmethod
    def _get_classes(self):
        """ implement self.classes """; ...
        
    @property
    def num_classes(self):
        return len(self.classes)

    def read(self, imagefile, label):
        if tf.__version__ >= '1.14.0':
            image = tf.io.decode_image(tf.io.read_file(imagefile))
        else:
            image = tf.image.decode_image(tf.io.read_file(imagefile))
        return image, label

    def preprocess(self, image, label):
        if self.resize_shape is not None:
            th, tw = self.resize_shape
            image = tf.image.resize_image_with_pad(image, th, tw)
            image.set_shape((th, tw, 3))

        if self.crop_shape is not None:
            image = tf.image.random_crop(image, (*self.crop_shape, 3))

        if self.rotate:
            rotate_fn = tf.keras.preprocessing.image.random_rotation
            rotate_fn = partial(rotate_fn, rg=45,
                                row_axis=0, col_axis=1, channel_axis=2)
            image = tf.py_func(rotate_fn, [image], x.dtype)

        if self.flip_left_right:
            image = tf.image.random_flip_left_right(image)
        if self.flip_up_down:
            image = tf.image.random_flip_up_down(image)

        image /= 255.0
        return image, label

    def preprocess_valid(self, image, label):
        if self.resize_shape is not None:
            th, tw = self.resize_shape
            image = tf.image.resize_image_with_pad(image, th, tw)
            image.set_shape((th, tw, 3))
        return image, label

    def _build(self):
        if self.train_or_test == 'train':
            idx = int(len(self) * (1 - self.validation_split))
            dataset = tf.data.Dataset.from_tensor_slices(self.samples)
            dataset = dataset.shuffle(len(self))
            train_set = dataset.take(idx)\
              .map(self.read)\
              .map(self.preprocess)\
              .prefetch(self.batch_size)\
              .batch(self.batch_size, drop_remainder=True)
            valid_set = dataset.skip(idx)\
              .map(self.read)\
              .map(self.preprocess_valid)\
              .prefetch(self.batch_size)\
              .batch(self.batch_size, drop_remainder=True)
            
            if tf.__version__ >= '2.0.0':
                self.train_loader = trainset
                self.valid_loader = validset
            else:
                self.train_iterator = train_set.make_one_shot_iterator()
                self.valid_iterator = valid_set.make_one_shot_iterator()
        else:
            raise NotImplementedError('Test mode is not implemented')
        

class DogsVsCats(Base):
    """ tf.data pipeline for Flower dataset
    https://www.kaggle.com/alxmamaev/flowers-recognition
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
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
                         validation_split=validation_split,
                         resize_shape=resize_shape,
                         crop_shape=crop_shape,
                         rotate=rotate,
                         flip_left_right=flip_left_right,
                         flip_up_down=flip_up_down)

    def _get_filenames(self):
        d = self.dataset_dir + '/' + self.train_or_test
        filepath_dog = d + '/dog*.jpg'
        filepath_cat = d + '/cat*.jpg'
        imagefiles_dog = glob(filepath_dog)
        imagefiles_cat = glob(filepath_cat)
        imagefiles =  imagefiles_dog + imagefiles_cat
        labels =  [0]*len(imagefiles_dog) + [1]*len(imagefiles_cat)
        self.samples = (imagefiles, labels)

    def _get_classes(self):
        self.classes = ['dog', 'cat']
        
