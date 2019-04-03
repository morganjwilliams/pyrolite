import unittest
from pathlib import Path
from pyrolite.util.general import *


class TestTempPath(unittest.TestCase):
    """
    Tests the temporary directory utility function.
    """

    def test_is_dir(self):
        self.assertTrue(temp_path().is_dir())


class TestFlattenDict(unittest.TestCase):
    """Tests flatten_dict utility function."""

    def test_simple(self):
        """Check that dictionaries are flattened."""
        D = dict(key0=dict(key2="b", key3=dict(key4="c")), key1="a")
        result = flatten_dict(D)
        # Check the dictionary contains no dictionaries
        self.assertFalse(any([isinstance(v, dict) for v in result.values()]))
        self.assertTrue(result["key4"] == "c")

    def test_multiple_equal_keys(self):
        """
        Checks results where there are multiple identical keys at different
        levels, using different climb parameters
        (priorise trunk vs leaf values).
        """
        expected = ["a", "c"]
        for ix, climb in enumerate([True, False]):
            with self.subTest(climb=climb):
                D = dict(key0=dict(key2="b", key3=dict(key1="c")), key1="a")
                result = flatten_dict(D, climb=climb)
                self.assertTrue(result["key1"] == expected[ix])


class TestSwapItem(unittest.TestCase):
    """Tests swap_item utility function."""

    def test_simple(self):
        """Checks that an item can be swapped."""
        L = ["0", 1, "a", "d", 4]

        for pull, push in [("a", 2)]:
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


if __name__ == "__main__":
    unittest.main()
