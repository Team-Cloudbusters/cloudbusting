from pathlib import Path
import random

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import downscale_local_mean
from tqdm import tqdm

import albumentations as albu
import cv2
from keras.utils import Sequence

from ..tools import rle_to_mask, mask_to_rle


class ImageSequence(Sequence):

    classes = ('Fish', 'Flower', 'Gravel', 'Sugar')

    def __init__(self, data_dir, image_shape=(1400, 2100), sample='train',
            batch_size=1, val_fraction=0.2, mask_downsample_factor=1,
            shuffle=True, seed=None):
        if sample not in ['train', 'val', 'test']:
            msg = "sample argument must be either 'train', 'val' or 'test'"
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
        self.random_state = np.random.RandomState(seed)

        self.images = self._get_images()
        self.labels = self._get_labels()
        self.transform = self._get_transform()

        self.set_index(val_fraction)
        self.on_epoch_end()

    def _get_images(self):
        if self.sample == 'test':
            image_dir = self.data_dir / 'test_images'
        else:
            image_dir = self.data_dir / 'train_images'

        files = sorted(image_dir.glob('*.jpg'))
        files = np.array(files)

        return files


    def _get_labels(self):
        if self.sample == 'test':
            return None

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
        files = self.images[idx]

        if self.sample == 'test':
            X = self._generate_test_data(files)
            return X, None

        else:
            X, Y = self._generate_train_data(files)
            return X, Y


    def set_index(self, val_fraction):

        index = np.arange(len(self.images))

        if self.sample == 'train':
            self.random_state.shuffle(index)
            i_split = int((1 - val_fraction) * index.size)
            index = index[:i_split]
            index = np.sort(index)

        elif self.sample == 'val':
            self.random_state.shuffle(index)
            i_split = int((1 - val_fraction) * index.size)
            index = index[i_split:]
            index = np.sort(index)

        self.index = index

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state.shuffle(self.index)


    def _generate_train_data(self, files):
        n_images = len(files)
        n_y_image, n_x_image = self.image_shape
        n_y_mask = n_y_image // self.mask_downsample_factor
        n_x_mask = n_x_image // self.mask_downsample_factor

        n_classes = len(self.classes)

        images = np.zeros([n_images, n_y_image, n_x_image, 3],
                    dtype=np.uint8)
        masks = np.zeros([n_images, n_y_mask, n_x_mask, n_classes],
                    dtype=bool)

        #seed for augmentation
        random.seed(self.random_state.rand())
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


    def _generate_test_data(self, files):
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


    def predict_labels(self, model, image_shape=None):
        if image_shape is None:
            n_y, n_x = self.image_shape
        else:
            n_y, n_x = image_shape

        transform = albu.Compose([
            #resize output
            albu.Resize(n_y, n_x, interpolation=cv2.INTER_LINEAR,
                    always_apply=True)
            ], p=1)

        label_names = []
        encodings = []

        for file in tqdm(self.images):
            if self.sample == 'test':
                X = self._generate_test_data([file])
            else:
                X, _ = self._generate_train_data([file])

            Y_pred = model.predict(X)

            out = transform(image=X[0], mask=Y_pred[0])
            mask = out['mask'] > 0.5

            image_name = file.name
            for i, class_ in enumerate(self.classes):
                image_label = '{}_{}'.format(image_name, class_)
                rle = mask_to_rle(mask[:,:,i])
                rle = ' '.join([str(i) for i in rle])

                label_names.append(image_label)
                encodings.append(rle)

        df = pd.DataFrame.from_dict({
                            'Image_Label': label_names,
                            'EncodedPixels': encodings})

        return df
