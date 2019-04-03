import unittest
import numpy as np
import pandas as pd
from pyrolite.util.types import *


class TestIscollection(unittest.TestCase):
    """Tests iscollection utility function."""

    def setUp(self):
        self.collections = [[1, 2], np.array([1, 2]), set([1, 2]), (1, 2)]
        self.notcollections = "a", "aa", 1, 1.2

    def test_collections(self):
        for obj in self.collections:
            with self.subTest(obj=obj):
                self.assertTrue(iscollection(obj))

    def test_not_collections(self):
        for obj in self.notcollections:
            with self.subTest(obj=obj):
                self.assertFalse(iscollection(obj))


if __name__ == "__main__":
    unittest.main()
