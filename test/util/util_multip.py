import unittest
import pandas as pd
import numpy as np
from pyrolite.util.multip import multiprocess, func_wrapper, combine_choices
import platform


def arbitary_function(**kwargs):
    """A function which will return what is passed."""
    return kwargs


class TestCombineChoices(unittest.TestCase):
    def setUp(self):
        self.choices = dict(A=[0, 1], B=["c", -1])

    def test_default(self):
        start = self.choices
        c = dict(start)
        out = combine_choices(c)
        self.assertEqual(c, start)  # don't change the grid
        self.assertTrue(len(out) == 4)

    def test_subset_uniuqe(self):
        start = dict(A=[0, 1, 0], B=[0])
        c = dict(start)  # only two unique values for A
        out = combine_choices(c)
        self.assertEqual(c, start)  # don't change the grid
        self.assertTrue(len(out) == 2)

    def test_no_choices(self):
        start = {}
        c = dict(start)
        out = combine_choices(c)
        self.assertEqual(c, start)  # don't change the grid
        self.assertTrue(len(out) == 1)


class TestFuncWrapper(unittest.TestCase):
    """Tests the argument-to-function expanstion utility function."""

    def test_construct(self):
        test_kwargs = dict(test_kwarg1=0)
        arg = (arbitary_function, test_kwargs)
        result = func_wrapper(arg)
        self.assertEqual(result, test_kwargs)


@unittest.skipUnless(
    platform.system() != "Windows", "Bug with multiprocessing testing on Windows"
)
class TestMultiprocess(unittest.TestCase):
    """Tests the multiprocess utility function."""

    def test_construct(self):
        test_params = [dict(test_kwarg1=0), dict(test_kwarg2=1)]
        results = multiprocess(arbitary_function, test_params)
        self.assertEqual(results, test_params)


if __name__ == "__main__":
    unittest.main()
