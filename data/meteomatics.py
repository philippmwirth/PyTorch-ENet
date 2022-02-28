import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils

import albumentations as A
import numpy as np
import torch
from PIL import Image


class Meteomatics(data.Dataset):
    """Dataset interface to the satellite and radar data.

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'central_europe'

    # Validation dataset root folders
    val_folder = 'north_america'

    # Test dataset root folders
    test_folder = 'mexico'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('unlabeled', (255, 255, 255)),
        ('0.00', (255, 255, 255)),
        ('0.04-0.07', (0, 0, 254)),
        ('0.07-0.15', (50, 101, 254)),
        ('0.15-0.30', (77, 153, 0)),
        ('0.30-0.60', (116, 196, 0)),
        ('0.60-1.20', (254, 203, 0)),
        ('1.20-2.40', (254, 152, 0)),
        ('2.40-2.80', (254, 0, 0 )),
        ('4.80-6.00', (153, 0, 0)),
    ])


    transforms = None

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.meteo_pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder, 'sat_ir'),
                extension_filter=self.img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_folder, 'target'),
                extension_filter=self.img_extension)

            FILTER: bool = False
            if FILTER:
                with open('Meteomatics_central_europe_50.txt') as f:
                    lines = f.readlines()
                    lines = [l.strip('\n') for l in lines]
                    lines = [l.strip('central_europe/') for l in lines]
                
                def _was_selected(string: str):
                    return any(string.endswith(l) for l in lines)

                self.train_data = [td for td in self.train_data if _was_selected(td)]
                self.train_labels = [tl for tl in self.train_labels if _was_selected(tl)]


        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder, 'sat_ir'),
                extension_filter=self.img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_folder, 'target'),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder, 'sat_ir'),
                extension_filter=self.img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_folder, 'target'),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.mode.lower() == 'train' and Meteomatics.transforms is not None:
            t = Meteomatics.transforms(image=np.array(img), mask=np.array(label))
            img, label = Image.fromarray(t['image']), Image.fromarray(t['mask'])

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
