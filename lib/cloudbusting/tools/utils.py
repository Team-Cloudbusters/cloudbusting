from pathlib import Path
import logging

import numpy as np
import pandas as pd
from skimage.measure import find_contours, approximate_polygon

__all__ = ['rle_to_mask', 'mask_to_paths']


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


def mask_to_paths(mask):
    # mask with zeros
    n_y, n_x = mask.shape
    new_shape = (n_y + 2, n_x + 2)
    mask_padded = np.zeros(new_shape, dtype=mask.dtype)
    mask_padded[1:-1,1:-1] = mask

    paths = find_contours(mask_padded, 0.5)

    approx_paths = []
    for path in paths:
        path = approximate_polygon(path, 0.25)
        path = np.array(path) - 1 #remove effect of padding
        approx_paths.append(path)

    return approx_paths

