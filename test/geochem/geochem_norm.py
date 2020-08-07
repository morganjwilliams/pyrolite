import unittest
import pyrolite
import numpy as np
from pyrolite.util.synthetic import normal_frame
from pyrolite.geochem.norm import (
    get_reference_files,
    update_database,
    Composition,
    get_reference_composition,
    all_reference_compositions,
)
from pyrolite.util.general import temp_path, remove_tempdir
from pyrolite.util.meta import pyrolite_datafolder


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.filename = get_reference_files()[-1]

    def test_default(self):
        C = Composition(self.filename)
        self.assertTrue(hasattr(C, "comp"))
        self.assertTrue(hasattr(C, "units"))
        self.assertTrue(hasattr(C, "units"))

    def test_describe(self):
        C = Composition(self.filename)
        desc = C.describe()
        self.assertIsInstance(desc, str)
        self.assertIn(C.name, desc)
        self.assertIn('doi', desc)

    def test_str(self):
        C = Composition(self.filename)
        s = str(C)
        self.assertIsInstance(s, str)
        self.assertIn("Model of", s)

    def test_repr(self):
        C = Composition(self.filename)
        s = repr(C)
        self.assertIn("Composition(", s)


class TestGetReferenceFiles(unittest.TestCase):
    def test_default(self):
        out = get_reference_files()
        self.assertIsInstance(out, list)
        self.assertTrue(len(out) > 5)
        self.assertIn("CH_PalmeONeill2014", [i.stem for i in out])


class TestGetAllReferenceCompositions(unittest.TestCase):
    def test_default(self):
        refs = all_reference_compositions()
        self.assertIsInstance(refs, dict)
        self.assertIn("Chondrite_PON", refs)


class TestGetReferenceComposition(unittest.TestCase):
    def test_default(self):
        rc = "Chondrite_PON"
        out = get_reference_composition(rc)
        self.assertIsInstance(out, Composition)


class TestUpdateReferenceDataBase(unittest.TestCase):
    def setUp(self):
        self.tmppath = temp_path(suffix="refdbtest")
        self.name = "refdb.json"
        self.path = self.tmppath / self.name
        if not self.tmppath.exists():
            self.tmppath.mkdir(parents=True)

    def test_default(self):
        update_database(path=self.path)
        self.assertTrue(self.path.exists())

    def tearDown(self):
        remove_tempdir(self.tmppath)


if __name__ == "__main__":
    unittest.main()
