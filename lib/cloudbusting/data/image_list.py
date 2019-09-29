import pandas as pd
from collections.abc import Sequence
from pathlib import Path

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

        self.labels = labels


class Image(object):

    def __init__(self, file, labels=None):
        self.file = file
        self.labels = labels
