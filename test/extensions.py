import unittest
import importlib
import pkgutil
import pyrolite.extensions
from pyrolite import load_extensions


class TestLoadExtensions(unittest.TestCase):
    def test_load_extensions(self):
        base = "pyrolite_"
        replace = ["util"]
        modules = {
            name.replace(base, ""): importlib.import_module(name)
            for finder, name, ispkg in pkgutil.iter_modules()
            if name.startswith(base)
        }
        load_extensions(base=base, replace=replace)
        current_extensions = dir(pyrolite.extensions)
        if len(modules):
            for name, module in modules.items():
                for r in replace:
                    name = name.replace(r, "")
                self.assertIn(name, current_extensions)
                self.assertEqual(getattr(pyrolite.extensions, name), module)


if __name__ == "__main__":
    unittest.main()
