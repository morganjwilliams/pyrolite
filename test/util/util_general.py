import unittest
from pathlib import Path
from pyrolite.util.general import (
    temp_path,
    flatten_dict,
    copy_file,
    remove_tempdir,
    swap_item,
    Timewith,
)


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

    def test_safemode(self):
        D = dict(key0=dict(key2="b", key3=dict(key4="c")), key1="a")
        result = flatten_dict(D, safemode=True)
        # Check the dictionary contains no dictionaries
        self.assertFalse(any([isinstance(v, dict) for v in result.values()]))
        self.assertTrue(result[("key0", "key3", "key4")] == "c")


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


class TestTimewith(unittest.TestCase):
    """Tests TimeWith utility class."""

    def setUp(self):
        pass

    def test_default(self):

        with Timewith(name="test_timer") as t:
            for i in range(2):
                print(t.elapsed)
        self.assertIn("Finished", t.checkpoints[-1])


if __name__ == "__main__":
    unittest.main()
