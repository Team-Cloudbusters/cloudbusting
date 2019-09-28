import numpy as np

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
