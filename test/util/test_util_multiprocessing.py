import unittest
import pandas as pd
import numpy as np
from pyrolite.util.multiprocessing import multiprocess, func_wrapper
import platform


def arbitary_function(**kwargs):
    """A function which will return what is passed."""
    return kwargs


class TestFuncWrapper(unittest.TestCase):
    """Tests the argument-to-function expanstion utility function."""

    def test_construct(self):
        test_kwargs = dict(test_kwarg1=0)
        arg = (arbitary_function, test_kwargs)
        result = func_wrapper(arg)
        self.assertEqual(result, test_kwargs)


class TestMultiprocess(unittest.TestCase):
    """Tests the multiprocess utility function."""

    def test_construct(self):
        test_params = [dict(test_kwarg1=0), dict(test_kwarg2=1)]
        if (__name__ == "__main__") or platform.system() != "Windows":
            # bug with multiprocessing testing on Windows
            results = multiprocess(arbitary_function, test_params)
            self.assertEqual(results, test_params)


if __name__ == "__main__":
    unittest.main()
