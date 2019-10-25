import numpy as np
from numpy.testing import assert_array_equal

from cloudbusting.tools import *


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


def test_mask_to_rle():
    #test1: null case
    mask = np.zeros([5,6], dtype=bool)

    out = mask_to_rle(mask)
    expected = []

    assert_array_equal(out, expected)

    #test2
    mask = np.zeros([5,6], dtype=bool)
    mask[:3,0] = True

    out = mask_to_rle(mask)
    expected = [1, 3]

    assert_array_equal(out, expected)

    #test3
    mask = np.zeros([5,6], dtype=bool)
    mask[:3,0] = True
    mask[:,2] = True

    out = mask_to_rle(mask)
    expected = [1, 3, 11, 5]

    assert_array_equal(out, expected)
