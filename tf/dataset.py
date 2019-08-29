import tensorflow as tf
from glob import glob
from abc import abstractmethod, ABCMeta
from functools import partial


class Base(metaclass=ABCMeta):
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
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
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.batch_size = batch_size

        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.rotate = rotate
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

        print('Building a dataset pipeline ...')
        self._get_filenames()
        self._get_classes()
        print('Found {} images.'.format(len(self.filenames)))
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

    def _build(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.samples)
        dataset = dataset.shuffle(len(self))\
          .map(self.read)\
          .map(self.preprocess)\
          .prefetch(self.batch_size)\
          .batch(self.batch_size, drop_remainder=True)
        if tf.__version__ >= '2.0.0':
            self.loader = dataset
        else:
            self.iterator = dataset.make_one_shot_iterator()

    def read(self, imagefile, label):
        if tf.__version__ >= '1.14.0':
            image = tf.io.decode(tf.io.read_file(imagefile))
        else:
            image = tf.image.decode_image(tf.io.read_file(imagefile))
        return image, label

    def preprocess(self, image, label):
        if self.resize_shape is not None:
            th, tw = self.resize_shape
            image = tf.image.resize_image_with_pad(image, tf, tw)
            image.set_shape(th, tw, 3)

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


class DogsVsCats(Base):
    """ tf.data pipeline for Flower dataset
    https://www.kaggle.com/alxmamaev/flowers-recognition
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
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
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         batch_size=batch_size,
                         resize_shape=resize_shape,
                         crop_shape=crop_shape,
                         rotate=rotate,
                         flip_left_right=flip_left_right,
                         flip_up_down=flip_up_down)

    def _get_filenames(self):
        d = self.dataset_dir + '/' + self.train_or_test

    @abstractmethod
    def _get_classes(self): ...
