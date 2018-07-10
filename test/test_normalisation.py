import unittest
import pyrolite
from pyrolite.normalisation import *


class TestScaleFunction(unittest.TestCase):
    """Tests scale function generator."""

    def setUp(self):
        self.df = pd.DataFrame()
        self.df['units'] = pd.Series(RELMASSS_UNITS.keys())
        self.df['values'] = pd.Series(np.random.rand(self.df.index.size))

    def test_same_units(self):
        """Checks exchange between values with the same units is unity."""
        for to in RELMASSS_UNITS.keys():
            with self.subTest(to=to):
                fm=to
                mult = scale_multiplier(fm, target_unit=to)
                self.assertFalse(np.isnan(mult))
                self.assertTrue(np.isclose(self.df['values'].values * mult,
                                           self.df['values'].values).all()
                               )

    def test_different_units(self):
        """Checks exchange between values with different units isn't unity."""
        fm_units = RELMASSS_UNITS.keys()
        to_units = RELMASSS_UNITS.keys()
        for to in to_units:
            for fm in [fu for fu in fm_units if not fu==to]:
                with self.subTest(fm=fm, to=to):
                    mult = scale_multiplier(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not RELMASSS_UNITS[to] == RELMASSS_UNITS[fm]:
                        self.assertFalse(np.isclose(
                                         self.df['values'].values * mult,
                                         self.df['values'].values).any()
                                         )

    @unittest.expectedFailure
    def test_failure_on_unknown_unit_in(self):
        """Checks the function raises when unknown units are used for "from"."""
        fm_units = ['notaunit', 'N/km2', 'm/s', 'ms-1']
        to_units = RELMASSS_UNITS.keys()
        for fm in fm_units:
            for to in to_units:
                with self.subTest(fm=fm, to=to):
                    mult = scale_multiplier(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not RELMASSS_UNITS[to] == RELMASSS_UNITS[fm]:
                        self.assertFalse(np.isclose(
                                         self.df['values'].values * mult,
                                         self.df['values'].values).any()
                                         )

    @unittest.expectedFailure
    def test_failure_on_unknown_unit_out(self):
        """Checks the function raises when unknown units are used for "to"."""
        fm_units = RELMASSS_UNITS.keys()
        to_units = ['notaunit', 'N/km2', 'm/s', 'ms-1']
        for fm in fm_units:
            for to in to_units:
                with self.subTest(fm=fm, to=to):
                    mult = scale_multiplier(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not RELMASSS_UNITS[to] == RELMASSS_UNITS[fm]:
                        self.assertFalse(np.isclose(
                                         self.df['values'].values * mult,
                                         self.df['values'].values).any()
                                         )


class TestRefcomp(unittest.TestCase):
    """Tests reference composition model."""

    def setUp(self):
        self.build_kwargs = dict(encoding='cp1252')
        pyrodir = os.path.realpath(pyrolite.normalisation.__file__)
        self.dir = (Path(pyrodir).parent /  "data" / "refcomp").resolve()
        assert self.dir.is_dir()
        self.files = [x for x in self.dir.iterdir() if x.is_file()]

        print(self.dir, self.files)
        self.CHfile = [f for f in self.files
                       if 'CH_PalmeONeill2014' in str(f)][0]

        self.cols = ['Ti', 'Mn', 'Cr', 'Ni']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_construction_with_dir(self):
        """Checks the model can build."""
        files = [x for x in self.dir.iterdir() if x.is_file()]
        for f in files:
            with self.subTest(f=f):
                refcomp = RefComp(f, **self.build_kwargs)

    def test_aggregate_oxides(self):
        """Checks the model can aggregate oxide components."""
        pass

    def test_collect_vars(self):
        """Checks that the model can assemble a list of relevant variables."""
        headers = ['Reservoir', 'Reference', 'ModelName', 'ModelType']
        CH = RefComp(self.CHfile, **self.build_kwargs)
        CH.collect_vars(headers=headers)
        self.assertTrue(hasattr(CH, 'vars'))
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


class TestReferenceCompositions(unittest.TestCase):
    """Tests the formation of a reference dictionary from a directory."""

    def setUp(self):
        self.build_kwargs = dict(encoding='cp1252')

    def test_build(self):
        """Checks that the dictionary constructs."""
        refdb = ReferenceCompositions(**self.build_kwargs)
        self.assertTrue(type(refdb) == dict)

    def test_content(self):
        """Checks that the dictionary structure is correct."""
        refdb = ReferenceCompositions(**self.build_kwargs)
        for k, v in refdb.items():
            self.assertTrue(type(v) == RefComp)


if __name__ == '__main__':
    unittest.main()
