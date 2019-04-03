import unittest
from pyrolite.util.web import *


class TestUrlify(unittest.TestCase):
    """
    Tests the urlify utility function.
    """

    def test_strip(self):
        for s in ["A B", "A_B", "A  ", "A B C D"]:
            with self.subTest(s=s):
                self.assertFalse(" " in urlify(s))


if __name__ == "__main__":
    unittest.main()
