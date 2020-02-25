import unittest
import matplotlib.path
import matplotlib.patches
from pyrolite.util.plot.interpolation import interpolated_patch_path

class InterpolatedPathPatch(unittest.TestCase):
    """
    Tests the interpolated_path_patch utility function.
    """

    def setUp(self):
        self.patch = matplotlib.patches.Ellipse((0, 0), 1, 2)

    def test_default(self):
        path = interpolated_patch_path(self.patch)
        self.assertTrue(isinstance(path, matplotlib.path.Path))

    def test_resolution(self):
        for res in [2, 10, 100]:
            with self.subTest(res=res):
                path = interpolated_patch_path(self.patch, resolution=res)
                self.assertTrue(path.vertices.shape[0] == res)


if __name__ == "__main__":
    unittest.main()
