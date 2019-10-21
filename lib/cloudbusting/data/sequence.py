from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import downscale_local_mean

import albumentations as albu
import cv2
from keras.utils import Sequence

from ..tools import rle_to_mask


class ImageSequence(Sequence):

    def __init__(self, data_dir, image_shape=(1400, 2100), sample='train',
            batch_size=1, mask_downsample_factor=1, shuffle=True):
        if sample not in ['train', 'test']:
            msg = "sample argument must be either 'train' or 'test'"
            raise Exception(msg)

        self.data_dir = Path(data_dir)
        self.image_shape = image_shape
        self.sample = sample
        self.batch_size = batch_size

        if (((image_shape[0] % mask_downsample_factor) == 0) and
            ((image_shape[1] % mask_downsample_factor) == 0)):
            self.mask_downsample_factor = mask_downsample_factor
        else:
            msg = ("image_shape {}, must be ivisible by"
                   "mask_downsample_factor {}")
            msg = msg.format(image_shape, mask_downsample_factor)
            raise Exception(msg)

        self.shuffle = shuffle

        self.images = self._get_images()
        self.labels = self._get_labels()
        if self.labels is not None:
            self.classes = tuple(self.labels.index.unique(level=1).to_list())
        else:
            self.classes = None
        self.transform = self._get_transform()

        self.index = np.arange(len(self.images))
        self.on_epoch_end()

    def _get_images(self):
        if self.sample == 'train':
            image_dir = self.data_dir / 'train_images'
        elif self.sample == 'test':
            image_dir = self.data_dir / 'test_images'

        files = sorted(image_dir.glob('*.jpg'))
        files = np.array(files)

        return files


    def _get_labels(self):
        if self.sample == 'test':
            return None

        #else sample == 'train'

        label_file = self.data_dir / 'train.csv'
        df = pd.read_csv(label_file)

        idx = df['Image_Label'].str.extract(r'(\w+).jpg_(\w+)')
        idx = pd.MultiIndex.from_frame(idx, names=['image_name', 'cloud_type'])

        labels = df['EncodedPixels']
        labels.name = 'encoded_pixels'
        labels.index = idx
        labels.loc[labels.isnull()] = ''

        return labels


    def __len__(self):
        return int(np.ceil(len(self.index) / self.batch_size))


    def __getitem__(self, i):

        i_l = i * self.batch_size
        i_r = (i+1) * self.batch_size
        idx = self.index[i_l:i_r]

        if self.sample == 'train':
            X, Y = self._generate_train_data(idx)
            return X, Y
        elif self.sample == 'test':
            X = self._generate_test_data(idx)
            return X, None


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index)


    def _generate_train_data(self, idx):
        files = self.images[idx]
        n_images = len(files)
        n_y_image, n_x_image = self.image_shape
        n_y_mask = n_y_image // self.mask_downsample_factor
        n_x_mask = n_x_image // self.mask_downsample_factor

        n_classes = len(self.classes)

        images = np.zeros([n_images, n_y_image, n_x_image, 3],
                    dtype=np.uint8)
        masks = np.zeros([n_images, n_y_mask, n_x_mask, n_classes],
                    dtype=bool)

        for i, file in enumerate(files):
            image, mask = self._load_data(file)

            #apply augmentations
            mask = mask.astype(np.uint8) #needed for opencv
            out = self.transform(image=image, mask=mask)

            mask = out['mask']
            #downsample mask if needed
            if self.mask_downsample_factor > 1:
                factor = self.mask_downsample_factor 
                mask = downscale_local_mean(mask, (factor, factor, 1)) >= 0.5

            images[i] = out['image']
            masks[i] = mask.astype(bool)

        return images, masks


    def _generate_test_data(self, idx):
        files = self.images[idx]
        n_images = len(files)
        n_y_image, n_x_image = self.image_shape

        images = np.zeros([n_images, n_y_image, n_x_image, 3],
                    dtype=np.uint8)

        for i, file in enumerate(files):
            image, _ = self._load_data(file)

            out = self.transform(image=image)

            images[i] = out['image']

        return images


    def _load_data(self, file):
        image = imread(file)

        if self.sample == 'test':
            return image, None

        image_name = file.stem

        n_y, n_x = image.shape[:2]
        n_classes = len(self.classes)

        mask = np.zeros([n_y, n_x, n_classes], dtype=bool)

        for i, class_ in enumerate(self.classes):
            encoding = self.labels[image_name, class_]
            encoding = [int(i) for i in encoding.split()]

            mask[:,:,i] = rle_to_mask(encoding, (n_y, n_x))

        return image, mask


    def _get_transform(self):

        n_y, n_x = self.image_shape

        transform = albu.Compose([
            #resize output
            albu.Resize(n_y, n_x, interpolation=cv2.INTER_AREA,
                    always_apply=True)
            ], p=1)

        return transform
