import unittest
from pathlib import Path
from pyrolite.util.general import *


class TestUrlify(unittest.TestCase):
    """
    Tests the urlify utility function.
    """
    def test_strip(self):
        for s in ['A B', 'A_B', 'A  ', 'A B C D']:
            with self.subTest(s=s):
                self.assertFalse(' ' in urlify(s))


class TestTempPath(unittest.TestCase):
    """
    Tests the temporary directory utility function.
    """

    def test_is_dir(self):
        self.assertTrue(temp_path().is_dir())


class TestIscollection(unittest.TestCase):
    """Tests iscollection utility function."""

    def setUp(self):
        self.collections = [[1, 2], np.array([1, 2]), set([1, 2]), (1, 2)]
        self.notcollections = 'a', 'aa', 1, 1.2

    def test_collections(self):
        for obj in self.collections:
            with self.subTest(obj=obj):
                self.assertTrue(iscollection(obj))

    def test_not_collections(self):
        for obj in self.notcollections:
            with self.subTest(obj=obj):
                self.assertFalse(iscollection(obj))


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


class TestCheckPerl(unittest.TestCase):
    """Tests the check for a working perl installation."""

    def test_check_perl(self):
        val = check_perl()
        self.assertTrue(isinstance(val, bool))


class TestCopyFile(unittest.TestCase):
    """Tests copy_file utility function."""

    def setUp(self):
        pass

    def test_simple(self):
        pass


class TestRemoveTempdir(unittest.TestCase):
    """Tests remove_tempdir utility function."""

    def setUp(self):
        pass

    def test_simple(self):
        pass


class TestExtractZip(unittest.TestCase):
    """Tests extract_zip utility function."""

    def setUp(self):
        pass

    def test_simple(self):
        pass


if __name__ == '__main__':
    unittest.main()
