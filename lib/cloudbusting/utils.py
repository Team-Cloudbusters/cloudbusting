from pathlib import Path
import logging

import numpy as np
import pandas as pd

def get_train_mask(image_id, cloud_type, dir_data):

    logger = logging.getLogger(__name__)

    dir_data = Path(dir_data)

    file = dir_data / 'train.csv'

    df = pd.read_csv(file, index_col='Image_Label')

    idx = '{}.jpg_{}'.format(image_id, cloud_type)
    rle = df.loc[idx,'EncodedPixels']

    if isinstance(rle, str):
        rle = [int(i) for i in rle.split()]
    else: #i.e. is NaN
        rle = []

    shape = (1400, 2100)
    mask = rle_to_mask(rle, shape)

    return mask


def rle_to_mask(rle, shape):
    """Convert run length encoded sequence to binary mask

    Parameters
    ----------
    rle : list of int
        Sequence of alternating offsets and lengths
        e.g. [264918, 937, 266318, 937]
    shape : (int, int)
        shape of original image

    Returns
    -------
    numpy.ndarray
        binary mask

    """

    rle = np.asarray(rle)

    offsets = rle[::2]
    lengths = rle[1::2]

    #start with transposed array
    mask = np.zeros(shape[::-1], dtype=bool)
    mask_flat = mask.ravel()

    for offset, length in zip(offsets, lengths):
        mask_flat[offset-1:offset+length-1] = True

    mask = mask.T

    return mask
