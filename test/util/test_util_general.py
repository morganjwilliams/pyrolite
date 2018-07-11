import unittest
import pandas as pd
import numpy as np
from pyrolite.util.general import *


class TestOnFinite(unittest.TestCase):
    """Tests on_finite array operation wrapper."""

    def test_inf(self):
        """Checks operations on inf values."""
        arr = np.array([0., 1., np.inf, -np.inf])

        for f in [np.min, np.max, np.mean]:
            with self.subTest(f=f):
                result = on_finite(arr, f)
                self.assertTrue(np.isclose(result, f(arr[:2])))

    def test_nan(self):
        """Checks operations on nan values."""
        arr = np.array([0., 1., np.nan, np.nan])

        for f in [np.min, np.max, np.mean]:
            with self.subTest(f=f):
                result = on_finite(arr, f)
                self.assertTrue(np.isclose(result, f(arr[:2])))


class TestFlattenDict(unittest.TestCase):
    """Tests flatten_dict utility function."""

    def test_simple(self):
        """Check that dictionaries are flattened."""
        D = dict(key0=dict(key2 = 'b',
                           key3 = dict(key4='c')),
                 key1='a')
        result = flatten_dict(D)
        # Check the dictionary contains no dictionaries
        self.assertFalse(any([isinstance(v, dict) for v in result.values()]))
        self.assertTrue(result['key4'] == 'c')

    def test_multiple_equal_keys(self):
        """
        Checks results where there are multiple identical keys at different
        levels, using different climb parameters
        (priorise trunk vs leaf values).
        """
        expected = ['a', 'c']
        for ix, climb in enumerate([True, False]):
            with self.subTest(climb=climb):
                D = dict(key0=dict(key2 = 'b',
                                   key3 = dict(key1='c')),
                         key1='a')
                result = flatten_dict(D, climb=climb)
                self.assertTrue(result['key1'] == expected[ix])


class TestSwapItem(unittest.TestCase):
    """Tests swap_item utility function."""

    def test_simple(self):
        """Checks that an item can be swapped."""
        L = ['0', 1, 'a', 'd', 4]

        for pull, push in [('a', 2)]:
            with self.subTest(pull=pull, push=push):
                result = swap_item(L, pull, push)
                self.assertTrue(len(result) == len(L))
                self.assertTrue(result[L.index(pull)] == push)

#swap_item

if __name__ == '__main__':
    unittest.main()
