import torch
import imageio
from torchvision import transforms
from abc import abstractmethod, ABCMeta


class Base(torch.utils.data.Dataset, metaclass=ABCMeta):
    """
    Abstract class to flexibly implement dataset pipeline
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 one_hot=True,
                 resize_shape=None,
                 crop_shape=None,
                 rotate=False,
                 flip_left_right=False,
                 flip_up_down=False):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - one_hot: boolean for one-hot encoding
          - resize_shape: tuple for resize shape (optional)
          - crop_shape: tuple for crop shape (optional)
          - rotate: boolean for rotation (optional)
          - flip_left_right: boolean for horizontal flip (optional)
          - flip_up_down: boolean for vertical flip (optional)
        """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.one_hot = one_hot
        self.is_validation = False

        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.rotate = rotate
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down

        print('Building a dataset pipeline ...')
        self._set_classes()
        self._get_filenames()
        print('Found {} images.'.format(len(self)))
        self._build_preprocess()
        self._build_transform()
        print('Done.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imagefile, label = self.samples[idx]
        image, label = self._read(imagefile, label)
        image = self.preprocess(image)
        if self.is_validation:
            return image, label
        image = self.transforms(image)
        return image, label

    def _read(self, imagefile, label):
        image = imageio.imread(imagefile)
        label = int(label)
        return image, label

    def _build_preprocess(self):
        preps = []
        # Implement preprocess (should be execute both train/val dataset)
        if self.resize_shape is not None:
            preps.append(transforms.Resize(self.resize_shape))
        preps.append(lambda x: x/255.0)
            
        self.preprocess = transforms.Compose(preps)

    def _build_transform(self):
        trans = []
        if self.crop_shape is not None:
            trans.append(transforms.RandomCrop(self.crop_shape))
        if self.rotate:
            trans.append(transforms.RandomRotation(45))
        if self.flip_left_right:
            trans.append(transforms.RandomHorizontalFlip())
        if self.flip_up_down:
            trans.append(transforms.RandomVerticalFlip())

        self.transform = transforms.Compose(trans)

        
