import unittest
import pyrolite
import numpy as np
from pyrolite.util.synthetic import test_df
from pyrolite.geochem.norm import (
    ReferenceCompositions,
    RefComp,
    get_reference_files,
    update_database,
    Composition,
    get_reference_composition,
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


class TestGetReferenceFiles(unittest.TestCase):
    def test_default(self):
        out = get_reference_files()
        self.assertIsInstance(out, list)
        self.assertTrue(len(out) > 5)
        self.assertIn("CH_PalmeONeill2014", [i.stem for i in out])


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


class TestRefcomp(unittest.TestCase):
    """Tests reference composition model."""

    def setUp(self):
        self.build_kwargs = dict(encoding="cp1252")
        self.dir = pyrolite_datafolder(subfolder="geochem") / "refcomp"
        assert self.dir.is_dir()
        self.files = [x for x in self.dir.iterdir() if x.is_file()]
        self.CHfile = [f for f in self.files if "CH_PalmeONeill2014" in str(f)][0]

        self.cols = ["Ti", "Mn", "Cr", "Ni"]
        self.df = test_df(cols=self.cols)

    def test_construction_with_dir(self):
        """Checks the model can build."""
        files = [x for x in self.dir.iterdir() if x.is_file()]
        for f in files:
            with self.subTest(f=f):
                refcomp = RefComp(f, **self.build_kwargs)

    def test_collect_vars(self):
        """Checks that the model can assemble a list of relevant variables."""
        headers = ["Reservoir", "Reference", "ModelName", "ModelType"]
        CH = RefComp(self.CHfile, **self.build_kwargs)
        CH.collect_vars(headers=headers)
        self.assertTrue(hasattr(CH, "vars"))
        self.assertTrue(type(CH.vars) == list)
        self.assertTrue(all([h not in CH.vars for h in headers]))

    def test_set_units(self):
        """Checks that the model can be represented as different units."""
        # Check function
        pass

    def test_set_units_reversible(self):
        """Checks that the unit conversion is reversible."""
        # Check reversible
        pass

    def test_normalize(self):
        """Checks that the model can be used for normalising a dataframe."""
        CH = RefComp(self.CHfile, **self.build_kwargs)
        norm = CH.normalize(self.df)
        # Test that type isn't changed
        self.assertTrue(type(norm) == type(self.df))

    def test_denormalize(self):
        """Checks that the model can be used for de-normalising a dataframe."""
        CH = RefComp(self.CHfile, **self.build_kwargs)
        norm = CH.normalize(self.df)
        unnorm = CH.denormalize(norm)
        # Test that type isn't changed
        self.assertTrue(type(unnorm) == type(self.df))
        self.assertTrue(np.allclose(unnorm.values.astype(float), self.df.values))

    def test_ratio_present(self):
        CH = RefComp(self.CHfile, **self.build_kwargs)
        for ratio in ["Mn/Cu"]:
            r = CH.ratio(ratio)
            self.assertTrue(np.isfinite(r))

    def test_getattr(self):
        CH = RefComp(self.CHfile, **self.build_kwargs)
        for attr in ["Mn", "Cu"]:
            a = getattr(CH, attr)
            self.assertTrue(np.isfinite(a))

    def test_repr(self):
        CH = RefComp(self.CHfile, **self.build_kwargs)
        self.assertTrue("RefComp" in repr(CH))

    def test_str(self):
        CH = RefComp(self.CHfile, **self.build_kwargs)
        self.assertTrue("Model of" in str(CH))


class TestReferenceCompositions(unittest.TestCase):
    """Tests the formation of a reference dictionary from a directory."""

    def setUp(self):
        self.build_kwargs = dict(encoding="cp1252")

    def test_build(self):
        """Checks that the dictionary constructs."""
        refdb = ReferenceCompositions(**self.build_kwargs)
        self.assertTrue(type(refdb) == dict)

    def test_content(self):
        """Checks that the dictionary structure is correct."""
        refdb = ReferenceCompositions(**self.build_kwargs)
        for k, v in refdb.items():
            self.assertTrue(type(v) == RefComp)


if __name__ == "__main__":
    unittest.main()
