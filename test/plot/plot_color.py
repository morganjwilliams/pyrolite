import numpy as np
import unittest
from pyrolite.plot.color import *


class TestProcessColor(unittest.TestCase):
    def setUp(self):
        self.black = (0.0, 0.0, 0.0, 1.0)
        self.multiblack = np.array([(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)])

    def test_hex(self):
        hex = "#000000"
        for c in [hex]:
            out = process_color(c=c)
            self.assertEqual(out.get("color"), self.black)
            self.assertTrue((out.get("c") == np.array([self.black])).all())

    def test_named(self):
        named = "black"
        for c in [named]:
            out = process_color(c=c)
            self.assertEqual(out.get("color"), self.black)
            self.assertTrue((out.get("c") == np.array([self.black])).all())

    def test_rgb(self):
        rgb = (0, 0, 0)
        for c in [rgb]:
            out = process_color(c=c)
            self.assertEqual(out.get("color"), self.black)
            self.assertTrue((out.get("c") == np.array([self.black])).all())

    def test_rgba(self):
        rgba = (0, 0, 0, 1)
        for c in [rgba]:
            out = process_color(c=c)
            self.assertEqual(out.get("color"), self.black)
            self.assertTrue((out.get("c") == np.array([self.black])).all())

    def test_hex_array(self):
        hex_list = ["#000000", "#000000"]
        hex_array = np.array(["#000000", "#000000"])
        for c in [hex_list, hex_array]:
            out = process_color(c=c)
            self.assertTrue((out.get("c") == self.multiblack).all())

    def test_named_array(self):
        named_list = ["black", "black"]
        named_array = np.array(["black", "black"])
        for c in [named_list, named_array]:
            out = process_color(c=c)
            self.assertTrue((out.get("c") == self.multiblack).all())

    def test_mixed_str_array(self):
        mixed_str_list = ["black", "#000000"]
        mixed_str_array = np.array(["black", "#000000"])
        for c in [mixed_str_list, mixed_str_array]:
            out = process_color(c=c)
            self.assertTrue((out.get("c") == self.multiblack).all())

    def test_rgb_array(self):
        rgb_list = [(0, 0, 0), (0, 0, 0)]
        rgb_array = np.array([(0, 0, 0), (0, 0, 0)])
        for c in [rgb_list, rgb_list]:
            out = process_color(c=c)
            self.assertTrue((out.get("c") == self.multiblack).all())

    def test_rgba_array(self):
        rgba_list = [(0, 0, 0, 1), ((0, 0, 0, 1))]
        rgba_array = np.array([(0, 0, 0, 1), (0, 0, 0, 1)])
        for c in [rgba_list, rgba_array]:
            out = process_color(c=c)
            self.assertTrue((out.get("c") == self.multiblack).all())

    def test_value_array(self):
        value_array = np.array([0.1, 0.9])
        for c in [value_array]:
            out = process_color(c=c)

    def test_categories(self):
        categories = ["Bird", "Fish", "Cat"]
        for c in [categories]:
            out = process_color(c=c)

    def test_categories_color_mappings(self):
        categories = ["Bird", "Fish", "Cat"]
        mappings = {"Bird": "green", "Fish": "blue", "Cat": "orange"}
        for c, m in zip([categories], [mappings]):
            out = process_color(c=c, color_mappings=m)

    @unittest.expectedFailure
    def test_singular_value(self):
        c = 1.0
        out = process_color(c=c)

    def test_mixed_input(self):
        c = ["0.5", (1, 0, 0), "black"]
        out = process_color(c=c)

    @unittest.expectedFailure
    def test_invalid_mixed_input(self):
        """Non-scaled floats etc."""
        c = [1.0, 10.1, "0.5", (1, 0, 0), "black"]
        out = process_color(c=c)

    def test_singular_with_alpha(self):
        alpha = 0.5
        for c in ["green"]:
            out = process_color(c=c, alpha=alpha)
            print(out["c"])
            self.assertTrue(np.allclose(out["c"][:, -1], alpha))

    def test_array_with_alpha(self):
        alpha = 0.5
        for c in [np.array([0.1, 0.9])]:
            out = process_color(c=c, alpha=alpha)
            self.assertTrue(np.allclose(out["c"][:, -1], alpha))


class TestGetCmode(unittest.TestCase):
    def test_hex(self):
        hex = ("#000000", "hex")
        for c, expect in [hex]:
            self.assertEqual(get_cmode(c), expect)

    def test_named(self):
        named = ("black", "named")
        for c, expect in [named]:
            self.assertEqual(get_cmode(c), expect)

    def test_rgb(self):
        rgb = ((0, 0, 0), "rgb")
        for c, expect in [rgb]:
            self.assertEqual(get_cmode(c), expect)

    def test_rgba(self):
        rgba = ((0, 0, 0, 1), "rgba")
        for c, expect in [rgba]:
            self.assertEqual(get_cmode(c), expect)

    def test_hex_array(self):
        hex_list = (["#000000", "#000000"], "hex_array")
        hex_array = (np.array(["#000000", "#000000"]), "hex_array")
        for c, expect in [hex_list, hex_array]:
            self.assertEqual(get_cmode(c), expect)

    def test_named_array(self):
        named_list = (["black", "black"], "named_array")
        named_array = (np.array(["black", "black"]), "named_array")
        for c, expect in [named_list, named_array]:
            self.assertEqual(get_cmode(c), expect)

    def test_mixed_str_array(self):
        mixed_str_list = (["black", "#000000"], "mixed_str_array")
        mixed_str_array = (np.array(["black", "#000000"]), "mixed_str_array")
        for c, expect in [mixed_str_list, mixed_str_array]:
            self.assertEqual(get_cmode(c), expect)

    def test_rgb_array(self):
        rgb_list = ([(0, 0, 0), (0, 0, 0)], "rgb_array")
        rgb_array = (np.array([(0, 0, 0), (0, 0, 0)]), "rgb_array")
        for c, expect in [rgb_list, rgb_list]:
            self.assertEqual(get_cmode(c), expect)

    def test_rgba_array(self):
        rgba_list = ([(0, 0, 0, 1), ((0, 0, 0, 1))], "rgba_array")
        rgba_array = (np.array([(0, 0, 0, 1), (0, 0, 0, 1)]), "rgba_array")
        for c, expect in [rgba_list, rgba_array]:
            self.assertEqual(get_cmode(c), expect)

    def test_value_array(self):
        value_array = (np.array([0.1, 0.9]), "value_array")
        for c, expect in [value_array]:
            self.assertEqual(get_cmode(c), expect)

    @unittest.expectedFailure
    def test_singular_value(self):
        c = 1.0
        cmode = get_cmode(c)

    def test_mixed_input(self):
        c = ["0.5", (1, 0, 0), "black"]
        cmode = get_cmode(c)
        self.assertEqual(cmode, 'mixed_fmt_color_array')

    @unittest.expectedFailure
    def test_invalid_mixed_input(self):
        """Non-scaled floats etc."""
        c = [1.0, 10.1, "0.5", (1, 0, 0), "black"]
        cmode = get_cmode(c)
        self.assertEqual(cmode, 'mixed_fmt_color_array')


if __name__ == "__main__":
    unittest.main()
