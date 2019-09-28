import numpy as np
from numpy.testing import assert_array_equal

from cloudbusting.utils import rle_to_mask


def test_rle_to_mask():
    #test1: null case
    rle = []
    shape = (5,6)

    out = rle_to_mask(rle, shape)
    expected = np.zeros([5,6], dtype=bool)

    assert_array_equal(out, expected)

    #test2
    rle = [1, 3]
    shape = (5,6)

    out = rle_to_mask(rle, shape)
    expected = np.zeros([5,6], dtype=bool)
    expected[:3,0] = True

    assert_array_equal(out, expected)

    #test3
    rle = [1, 3, 11, 5]
    shape = (5,6)

    out = rle_to_mask(rle, shape)
    expected = np.zeros([5,6], dtype=bool)
    expected[:3,0] = True
    expected[:,2] = True

    assert_array_equal(out, expected)
