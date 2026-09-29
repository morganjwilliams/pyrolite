import unittest
from pathlib import Path

from pyrolite.util.meta import (
    pyrolite_datafolder,
)


class TestPyroliteDataFolder(unittest.TestCase):
    def test_default(self):
        folder = pyrolite_datafolder()
        self.assertIsInstance(folder, Path)
        self.assertTrue(folder.exists())
        self.assertTrue(folder.is_dir())

    def test_subfolders(self):
        for subf in ["Aitchison", "geochem", "models", "radii", "timescale"]:
            with self.subTest(subf=subf):
                folder = pyrolite_datafolder(subfolder=subf)
                self.assertIsInstance(folder, Path)
                self.assertTrue(folder.exists())
                self.assertTrue(folder.is_dir())


# ToLogger

# take_me_to_the_docs

# sphinx_doi_link

# subkwargs

# inargs

# get_additional_params

# update_docstring_references

if __name__ == "__main__":
    unittest.main()
