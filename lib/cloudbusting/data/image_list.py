import pandas as pd
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ..tools import rle_to_mask, mask_to_paths

__all__ = ['TrainImageList']

class ImageList(Sequence):
    def __init__(self, dir_data):
        self.dir_data = Path(dir_data)
        self.image_names = []
        self.labels = None


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, i):
        if isinstance(i, int):
            image_name = self.image_names[i]

        if isinstance(i, str):
            if i in self.image_names:
                image_name = i
            else:
                raise KeyError()

        file = self.dir_images / '{}.jpg'.format(image_name)
        if self.labels is not None:
            labels = self.labels.loc[image_name]

        return Image(file, labels=labels)


    def _load_image_names(self):
        matches = self.dir_images.glob('*.jpg')

        for match in matches:
            image_name = match.stem
            self.image_names.append(image_name)


class TrainImageList(ImageList):

    def __init__(self, dir_data):
        super().__init__(dir_data)

        self.dir_images = self.dir_data / 'train_images'
        self._load_image_names()
        self._load_train_labels()


    def _load_train_labels(self):
        file_train = self.dir_data / 'train.csv'
        df = pd.read_csv(file_train)
        idx = df['Image_Label'].str.extract(r'(\w+).jpg_(\w+)')
        idx = pd.MultiIndex.from_frame(idx, names=['image_name', 'cloud_type'])

        labels = df['EncodedPixels']
        labels.name = 'encoded_pixels'
        labels.index = idx
        labels.loc[labels.isnull()] = ''

        self.labels = labels


class Image(object):

    cloud_types = ('Fish', 'Flower', 'Gravel', 'Sugar')

    def __init__(self, file, labels=None):
        file = Path(file)
        self.file = file
        self.name = file.stem

        self._data = None #lazy load
        self._masks = None #create when needed

        #convert pandas series to dict
        if isinstance(labels, pd.Series):
            labels = labels.to_dict()
        if labels is not None:
            for k, v in labels.items():
                labels[k] = [int(i) for i in v.split()]
        self.labels = labels


    @property
    def data(self):
        if self._data is None:
            data = imread(self.file)
            self._data = data
        else:
            data = self._data
        return data


    @property
    def shape(self):
        return self.data.shape[:2]


    @property
    def masks(self):
        if self.labels is None:
            masks = self._masks

        if self._masks is None:
            shape = self.shape + (len(self.cloud_types),)
            masks = np.zeros(shape, dtype=bool)

            for i, cloud_type in enumerate(self.cloud_types):
                encoding = self.labels[cloud_type]
                masks[:,:,i] = rle_to_mask(encoding, self.shape)

            self._masks = masks

        else:
            masks = self._masks

        return masks


    def _plot_masks(self, ax):
        colors = {
                'Fish': 'tab:blue',
                'Flower': 'tab:orange',
                'Gravel': 'tab:green',
                'Sugar': 'tab:red',
                }

        masks = self.masks
        if masks is None: #do nothing
            return

        for i, cloud_type in enumerate(self.cloud_types):
            mask = masks[:,:,i]
            paths = mask_to_paths(mask)

            color = colors[cloud_type]

            for path in paths:
                # swap x,y
                path = path[:,::-1]
                patch = Polygon(path, closed=True, facecolor='none',
                        edgecolor=color)
                ax.add_patch(patch)

            #add text label to approx top left corner
            if np.sum(mask):
                x0 = np.argmax(np.max(mask, axis=0))
                y0 = np.argmax(mask[:,x0])

                t = ax.text(x0+10, y0+10, cloud_type, color=color,
                        va='top', ha='left', backgroundcolor='#ffffffaa')
                t._bbox_patch.set_boxstyle('Square, pad=0.0')


    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        ax.imshow(self.data)
        ax.set_title(self.name)

        self._plot_masks(ax)

        return ax
