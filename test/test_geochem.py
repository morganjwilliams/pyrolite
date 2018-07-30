import unittest
import pandas as pd
import numpy as np
from pyrolite.geochem import *
from pyrolite.normalisation import ReferenceCompositions


def test_df(cols=['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2'],
            index_length=10):
    return pd.DataFrame({k: v for k,v in zip(cols,
                         np.random.rand(len(cols), index_length))})


def test_ser(index=['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']):
    return pd.Series({k: v for k,v in zip(index, np.random.rand(len(index)))})


class TestGetRadii(unittest.TestCase):
    """Checks the radii getter."""

    def setUp(self):
        self.ree = REE()

    def test_ree_radii(self):
        radii = get_radii(self.ree[0])
        self.assertTrue(isinstance(radii, float))

    def test_ree_radii_list(self):
        radii = get_radii(self.ree)
        self.assertTrue(isinstance(radii, list))


class TestIsChem(unittest.TestCase):
    """Checks the 'is a chem' function."""

    def setUp(self):
        self.ree = REE()

    def test_ischem_str(self):
        ret = ischem(self.ree[0])
        self.assertTrue(isinstance(ret, bool))
        self.assertTrue(ret)

    def test_notchem_str(self):
        ret = ischem("Notachemical")
        self.assertTrue(isinstance(ret, bool))
        self.assertFalse(ret)

    def test_ischem_list(self):
        ret = ischem(self.ree)
        self.assertTrue(isinstance(ret, list))
        self.assertTrue(all([isinstance(i, bool) for i in ret]))

class TestToChem(unittest.TestCase):
    """Checks the 'transform to chem' function."""

    def setUp(self):
        self.ree = REE()

    def test_tochem_str(self):
        ret = tochem([self.ree[0]])
        self.assertTrue(ret == [str(self.ree[0])])

    def test_tonotchem_str(self):
        ret = tochem(["Notachemical"])
        self.assertTrue(ret == ["Notachemical"])

    def test_tochem_list(self):
        ret = tochem(self.ree)
        self.assertTrue(ret == list(map(str, self.ree)))


class TestToMolecular(unittest.TestCase):
    """Tests pandas molecular conversion operator."""

    def setUp(self):
        self.df = test_df()

    def test_single(self):
        """Checks results on single records."""
        ret = to_molecular(self.df.head(1))
        self.assertTrue((ret != self.df.head(1)).all().all())

    def test_multiple(self):
        """Checks results on multiple records."""
        ret = to_molecular(self.df)
        self.assertTrue((ret != self.df).all().all())

class TestToWeight(unittest.TestCase):
    """Tests pandas weight conversion operator."""

    def setUp(self):
        self.df = test_df()

    def test_single(self):
        """Checks results on single records."""
        ret = to_weight(self.df.head(1))
        self.assertTrue((ret != self.df.head(1)).all().all())

    def test_multiple(self):
        """Checks results on multiple records."""
        ret = to_weight(self.df)
        self.assertTrue((ret != self.df).all().all())


class TestWeightMolarReversal(unittest.TestCase):
    """Tests the reversability of weight-molar unit transformations."""

    def setUp(self):
        self.df = pd.DataFrame({'MgO':20.0, 'SiO2':30.0, 'K2O':5.0, 'Na2O':2.0},
                               index=[0])
        self.components = ['MgO', 'SiO2', 'K2O']

    def test_weightmolar_reversal(self):
        """
        Tests reversability of the wt-mol conversions.
        Examines differences between dataframes, and
        asserts that any discrepency is explained by np.nan components
        (and hence not actual differences).
        """
        renorm = False
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                M = to_molecular(self.df.loc[:, self.components],
                                                  renorm=renorm)
                W = to_weight(self.df.loc[:, self.components],
                                                  renorm=renorm)

                W_M = to_weight(M, renorm=renorm)
                M_W = to_molecular(W, renorm=renorm)

                # Where values are not close, it's because of nans
                original = self.df.loc[:, self.components]
                if renorm:
                    original = renormalise(original, components=self.components)
                W_M_close = np.isclose(W_M.values, original.values)
                self.assertTrue(np.isnan(W_M.values[~W_M_close]).all())

                M_W_close = np.isclose(M_W.values, original.values)
                self.assertTrue(np.isnan(M_W.values[~M_W_close]).all())


class TestGetCations(unittest.TestCase):
    """Tests the cation calculator."""

    def test_none(self):
        """Check the function works for no cations."""
        for cationstring in ['O', 'O2', '',]:
            with self.subTest(cationstring=cationstring):
                self.assertTrue(len(get_cations(cationstring))==0)

    def test_single(self):
        """Check the function works for a single cation."""
        for cationstring in ['SiO2', 'MgO', 'Si',]:
            with self.subTest(cationstring=cationstring):
                self.assertTrue(len(get_cations(cationstring))==1)

    def test_multiple(self):
        """Check the function works for multiple cations."""
        for cationstring in ["MgSiO2","MgSO4", "CaCO3",
                             "Na2Mg3Al2Si8O22(OH)2",]:
            with self.subTest(cationstring=cationstring):
                self.assertTrue(len(get_cations(cationstring))>1)

    def test_exclude(self):
        """Checks that the exclude function works."""
        for ox, excl in [('MgO', ['O']),
                         ('MgO', []),
                         ('MgSO4', ['O', 'S']),
                         ('MgSO4', ['S']),
                         ("Mg(OH)2", ['O', 'H']),
                         ("Mg(OH)2", ['H'])]:
            with self.subTest(ox=ox, excl=excl):
                self.assertTrue(len(get_cations(ox, exclude=excl))==1)


class TestCommonElements(unittest.TestCase):
    """Tests the common element generator."""

    def test_cutoff(self):
        """Check the function works normal cutoff Z numbers."""
        for cutoff in [1, 15, 34, 63, 93]:
            with self.subTest(cutoff=cutoff):
                self.assertTrue(common_elements(output='formula',
                                                cutoff=cutoff
                                                )[-1].number==cutoff)

    def test_high_cutoff(self):
        """Check the function works silly high cutoff Z numbers."""
        for cutoff in [119, 1000, 10000]:
            with self.subTest(cutoff=cutoff):
                self.assertTrue(len(common_elements(output='formula',
                                                    cutoff=cutoff))<130)
                self.assertTrue(common_elements(output='formula',
                                                cutoff=cutoff
                                                )[-1].number<cutoff)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for el in common_elements(cutoff=10, output='formula'):
            with self.subTest(el=el):
                self.assertIs(type(el), type(pt.elements[0]))

    def test_string_output(self):
        """Check the function produces string output."""
        for el in common_elements(cutoff=10, output='string'):
            with self.subTest(el=el):
                self.assertIs(type(el), str)


class TestREE(unittest.TestCase):
    """Tests the Rare Earth Element generator."""

    def setUp(self):
        self.min_z = 57
        self.max_z = 71

    def test_complete(self):
        """Check all REE are present."""
        reels = REE(output='formula')
        ns = [el.number for el in reels]
        for n in range(self.min_z, self.max_z + 1):
            with self.subTest(n=n):
                self.assertTrue(n in ns)

    def test_precise(self):
        """Check that only the REE are returned."""
        reels = REE(output='formula')
        ns = [el.number for el in reels]
        self.assertTrue(min(ns) == self.min_z)
        self.assertTrue(max(ns) == self.max_z)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for el in REE(output='formula'):
            with self.subTest(el=el):
                self.assertIs(type(el), type(pt.elements[0]))

    def test_string_output(self):
        """Check the function produces string output."""
        for el in REE(output='string'):
            with self.subTest(el=el):
                self.assertIs(type(el), str)

    def test_include_extras(self):
        """Check the ability to add extra elements such as Y."""
        pass


class TestSimpleOxides(unittest.TestCase):
    """Tests the simple oxide generator."""

    @unittest.expectedFailure
    def test_none(self):
        """Check the function returns no oxides for no elements in."""
        simple_oxides('', output='formula')

    def test_one(self):
        """Check the function returns oxides for one element in."""
        self.assertTrue(len(simple_oxides('Si', output='formula'))>=1)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for ox in simple_oxides('Si', output='formula'):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), type(pt.formula('SiO2')))

    def test_string_output(self):
        """Check the function produces string output."""
        for ox in simple_oxides('Si', output='string'):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), str)


class TestCommonOxides(unittest.TestCase):
    """Tests the common oxide generator."""

    def test_none(self):
        """Check the function returns no oxides for no elements in."""
        # When not passed elements, this function uses elements up to Uranium
        # to generate oxides instead.
        out = common_oxides(elements=[], output='formula')
        self.assertTrue(len(out) != 1)

    def test_one(self):
        """Check the function returns oxides for one element in."""
        els = ['Si']
        out = common_oxides(elements=els, output='formula')
        self.assertTrue(len(out) >= 1)
        for ox in out:
            with self.subTest(ox=ox):
                # All oxides are from elements contained in the list
                self.assertIn(get_cations(ox)[0].__str__(), els)

    def test_multiple(self):
        """Check the function returns oxides for muliple elements in."""
        els = ['Si', 'Mg', 'Ca']
        out = common_oxides(elements=els, output='formula')
        self.assertTrue(len(out) >= len(els))
        for ox in out:
            with self.subTest(ox=ox):
                # All oxides are from elements contained in the list
                self.assertIn(get_cations(ox)[0].__str__(), els)

    @unittest.expectedFailure
    def test_invalid_elements(self):
        """Check the function fails for invalid input."""
        not_els = [['SiO2'], ['notanelement'], ['Ci']]
        for els in not_els:
            with self.subTest(els=els):
                common_oxides(elements=els, output='formula')

    def test_formula_output(self):
        """Check the function produces formula output."""
        for ox in common_oxides(output='formula'):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), type(pt.formula('SiO2')))

    def test_string_output(self):
        """Check the function produces string output."""
        for ox in common_oxides(output='string'):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), str)

    def test_addition(self):
        """Checks the addition functionality."""
        pass

    # As stands, unless addition == [], for string output extras are returned
    def test_precise(self):
        """Check that only relevant oxides are returned."""
        for els in [['Li'], ['Ca', 'Ti'], ['Li', 'Mg', 'K']]:
            with self.subTest(els=els):
                for ox in common_oxides(elements=els, output='formula'):
                    # All oxides are from elements contained in the list
                    self.assertIn(get_cations(ox)[0].__str__(), els)


class TestDevolatilise(unittest.TestCase):
    """Tests the devolatilisation transformation."""

    def setUp(self):
        self.cols = ['SiO2', 'K2O', 'H2O', 'H2O_PLUS','LOI']
        self.one_row = np.array([[40., 3., 5., 0.1, 7.]])
        self.two_rows = np.array([[40., 3., 5., 0.1, 7.],
                                  [40., 3., 5., 0.1, 7.]])

    def test_none(self):
        """Check the function copes with no records."""
        df = pd.DataFrame(data=None, columns=self.cols)
        out = devolatilise(df)
        self.assertTrue(out is not None)
        self.assertIs(type(out), pd.DataFrame)
        self.assertEqual(out.index.size, 0)

    def test_one(self):
        """Check the transformation functions for one record."""
        df = pd.DataFrame(data=self.one_row)
        df.columns =  self.cols
        self.assertEqual(devolatilise(df).index.size, 1)

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        df = pd.DataFrame(data=self.two_rows)
        df.columns =  self.cols
        self.assertEqual(devolatilise(df).index.size, self.two_rows.shape[0])

    def test_renorm(self):
        """Checks closure is achieved when renorm is used."""
        df = pd.DataFrame(data=self.two_rows)
        df.columns =  self.cols
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                devdf = devolatilise(df, renorm=renorm)
                equality = (devdf.values == \
                            df.loc[:, devdf.columns].values).all()
                # For renorm = True, values will not be the same
                # For renorm = False, values should be the same
                self.assertTrue(equality != renorm)
                if renorm:
                    self.assertTrue((devdf.sum(axis=1) == 100.).all())

    def test_exclude_precise(self):
        """Checks that exclusion occurrs correctly."""
        exclude = ['H2O', 'H2O_PLUS', 'H2O_MINUS', 'CO2', 'LOI']
        df = pd.DataFrame(data=self.one_row)
        df.columns =  self.cols
        devdf = devolatilise(df, exclude=exclude)
        # There should be all those which weren't excluded
        self.assertTrue(np.array([i in devdf.columns for i in [i for i in df.columns if i not in exclude]]).all())
        # There should be no new things which where unexpected included
        self.assertTrue(np.array([i in df.columns for i in devdf.columns]).all())


class TestOxideConversion(unittest.TestCase):
    """Tests the pandas oxide conversion function generator."""

    def test_string_input(self):
        """Check that the function accepts string formatted inputs."""
        oxin, oxout = 'Fe', 'FeO'
        self.assertTrue(oxide_conversion(oxin, oxout) is not None)

    def test_formula_input(self):
        """Check that the function accepts formula formatted inputs."""
        oxin, oxout = pt.formula('Fe'), pt.formula('FeO')
        self.assertTrue(oxide_conversion(oxin, oxout) is not None)

    def test_different_inputs(self):
        """Check that the function accepts two different formats of inputs."""
        oxin, oxout = pt.formula('Fe'), 'FeO'
        self.assertTrue(oxide_conversion(oxin, oxout) is not None)

    def test_function_generation(self):
        """Check that a vaild function is returned."""
        oxin, oxout = 'Fe', 'FeO'
        f =  oxide_conversion(oxin, oxout)
        self.assertTrue(callable(f))

    def test_function_docstring(self):
        """Check the function docstring includes the oxide info."""
        oxin, oxout = pt.formula('Fe'), pt.formula('FeO')
        for oxin, oxout in [(pt.formula('Fe'), pt.formula('FeO')),
                            ('Fe', 'FeO'),
                            (pt.formula('Fe'), 'FeO')]:
            with self.subTest(oxin=oxin, oxout=oxout):
                f =  oxide_conversion(oxin, oxout)
                doc = f.__doc__
                self.assertTrue((str(oxin) in doc) and (str(oxin) in doc))
                self.assertTrue('{} to {}'.format(oxin, oxout) in doc)

    def test_same(self):
        """Check the function retains unit for the same in-out."""
        oxin, oxout = 'FeO', 'FeO'
        ser = pd.Series([1., 1.])
        f = oxide_conversion(oxin, oxout)
        self.assertTrue((f(ser) == ser).all())

    def test_multiple_cations(self):
        """Check the function works for multiple-cation simple oxides."""
        oxin, oxout = 'FeO', 'Fe2O3'
        ser = pd.Series([1., 1.])
        f = oxide_conversion(oxin, oxout)
        # Add oxygen, gains mass
        self.assertTrue((f(ser) >= ser).all())

    def test_oxidise(self):
        """Check the function works for oxidation."""
        oxin, oxout = 'FeO', 'Fe'
        ser = pd.Series([1., 1.])
        f =  oxide_conversion(oxin, oxout)
        # Lose oxygen, gains mass
        self.assertTrue((f(ser) <= ser).all())

    def test_reduce(self):
        """Check the function works for reduction."""
        oxin, oxout = 'Fe', 'FeO'
        ser = pd.Series([1., 1.])
        f =  oxide_conversion(oxin, oxout)
        # Add oxygen, gains mass
        self.assertTrue((f(ser) >= ser).all())

    def test_molecular(self):
        """Check that the generated function can convert molecular data."""
        oxin, oxout = 'Fe', 'FeO'
        ser = pd.Series([1., 1.])
        f =  oxide_conversion(oxin, oxout)
        # Same number of atoms in each = should be same
        self.assertTrue((f(ser, molecular=True) == ser).all())

        oxin, oxout = 'Fe', 'Fe2O3'
        ser = pd.Series([1., 1.])
        f =  oxide_conversion(oxin, oxout)
        # Twice number of atoms in Fe2O3, should be half the moles of Fe2O3
        self.assertTrue((f(ser, molecular=True) == (0.5 * ser)).all())

    @unittest.expectedFailure
    def test_different_cations(self):
        """Check that the function fails on receiving different elements."""
        oxin, oxout = 'Fe', 'NiO'
        f =  oxide_conversion(oxin, oxout)


class TestRecalculateRedox(unittest.TestCase):
    """Tests the pandas dataframe redox conversion."""

    def setUp(self):
        self.cols = 'FeO', 'Fe2O3', 'Fe2O3T'
        self.two_rows = np.array([[0.5, 0.3, 0.1],
                                  [0.5, 0.3, 0.1]])
        self.df = pd.DataFrame(self.two_rows, columns=self.cols)

    def test_none(self):
        """Check the function copes with no records."""
        df = self.df.head(0)
        out = recalculate_redox(df)
        self.assertTrue(out is not None)
        self.assertIs(type(out), pd.DataFrame)
        self.assertEqual(out.index.size, 0)

    def test_one(self):
        """Check the transformation functions for one record."""
        df = self.df.head(1)
        self.assertEqual(recalculate_redox(df).index.size,
                         df.index.size)

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        df = self.df
        self.assertEqual(recalculate_redox(df).index.size,
                         df.index.size)

    def test_to_oxidised(self):
        """Check the oxidised form is returned when called."""
        df = self.df
        recalculate_redox(df, to_oxidised=True)

    def test_to_reduced(self):
        """Check the reduced form is returned when called."""
        df = self.df
        recalculate_redox(df, to_oxidised=False)

    def test_renorm(self):
        """Checks closure is achieved when renorm is used."""
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                reddf = recalculate_redox(self.df, renorm=renorm)
                if renorm:
                    self.assertTrue((reddf.sum(axis=1) == 100.).all())
                else:
                    c = [i for i in reddf.columns if i in self.df.columns]
                    equality = (reddf.loc[:, c].values ==
                                self.df.loc[:, c].values)
                    self.assertTrue(equality.all())


    def test_total_suffix(self):
        """Checks that different suffixes can be used."""
        pass

    def test_columns_dropped(self):
        """Checks that only one redox state is found in output."""
        pass

class TestAggregateCation(unittest.TestCase):
    """Tests the pandas dataframe cation aggregation transformation."""

    def setUp(self):
        self.cols = ['MgO', 'FeO', 'Fe2O3', 'Mg', 'Fe', 'FeOT']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_none(self):
        """Check the transformation copes with no records."""
        df = self.df.head(0).copy()
        for cation in ['Mg', 'Fe']:
            with self.subTest(cation=cation):
                aggdf = aggregate_cation(df, cation)
                # Check that only one form is returned

    def test_one(self):
        """Check the transformation functions for one record."""
        df = self.df.head(1).copy()
        for cation in ['Mg', 'Fe']:
            with self.subTest(cation=cation):
                aggdf = aggregate_cation(df, cation)
                # Check that only one form is returned

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        df = self.df.copy()
        for cation in ['Mg', 'Fe']:
            with self.subTest(cation=cation):
                aggdf = aggregate_cation(df, cation)
                # Check that only one form is returned

    def test_oxide_return(self):
        """Checks that oxide forms are returned."""
        df = self.df.head(1).copy()
        cation='Mg'
        aggdf = aggregate_cation(df, cation, form='oxide')
        # Check presence

        # Check absence of others

        # Check preciseness

        # Check no additional features

    def test_element_return(self):
        """Checks that element forms are returned."""
        df = self.df.head(1).copy()
        cation='Mg'
        aggdf = aggregate_cation(df, cation, form='element')
        # Check presence

        # Check absence of others

        # Check preciseness

        # Check no additional features

    def check_unit_scale(self):
        """Checks that the unit scales are used."""
        df = self.df.head(1)
        cation='Mg'
        for unit_scale in [0.1, 10, 10000]:
            with self.subTest(unit_scale=unit_scale):
                aggdf = aggregate_cation(df, cation, unit_scale=unit_scale)


class TestMultipleCationInclusion(unittest.TestCase):
    """Tests the pandas dataframe multiple inclusion checking."""

    def setUp(self):
        self.cols = ['MgO', 'FeO', 'Fe2O3', 'Mg', 'Fe', 'FeOT']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_none(self):
        """Check the function copes with no records."""
        # Note that this function runs from columns - doesn't need records
        df = self.df.head(0)
        self.assertTrue(len(
                        check_multiple_cation_inclusion(df)
                        )>0)
        self.assertTrue(all([i.__str__() in ['Mg', 'Fe'] for i in
                        check_multiple_cation_inclusion(df)]))

    def test_one(self):
        """Check the transformation functions for one record."""
        df = self.df.head(1)
        self.assertTrue(len(
                        check_multiple_cation_inclusion(df)
                        )>0)
        self.assertTrue(all([i.__str__() in ['Mg', 'Fe'] for i in
                        check_multiple_cation_inclusion(df)]))

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        df = self.df
        self.assertTrue(len(
                        check_multiple_cation_inclusion(df)
                        )>0)
        self.assertTrue(all([i.__str__() in ['Mg', 'Fe'] for i in
                        check_multiple_cation_inclusion(df)]))

    def test_exclusion(self):
        """Checks that exclusions are properly handled."""
        # Check that excluded components aren't considered
        pass

    def test_output(self):
        """Checks that the list returned is complete."""

        # Check complete

        # Check precise

        pass


class TestAddRatio(unittest.TestCase):
    """Tests the ratio addition."""

    def setUp(self):
        self.df = pd.DataFrame(columns=['Si', 'Mg', 'MgO', 'CaO'])

    def test_none(self):
        """Check the ratio addition copes with no records."""
        pass

    def test_one(self):
        """Check the ratio addition for one record."""
        pass

    def test_valid_ratios(self):
        """Check the addition works for valid pairs."""
        pass

    @unittest.expectedFailure
    def test_invalid_ratios(self):
        """Check the addition fails for invalid pairs."""
        df = self.df.copy()
        for ratio in ['Ca/Si', # Ca not present
                      'K.Na', # Invalid format
                      'Mg/Si/MgO' # multiple delimiters
                      ]:
              with self.subTest(ratio=ratio):
                  add_ratio(df, ratio=ratio)

    def test_alias(self):
        """Check that aliases can be used."""
        pass

    def test_convert(self):
        """Check that lambda conversion works."""
        pass


class TestAddMgNo(unittest.TestCase):
    """Tests the MgNo addition."""

    def setUp(self):
        self.cols = ['MgO', 'FeO', 'Fe2O3', 'Mg', 'Fe', 'FeOT']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_none(self):
        """Check the ratio addition copes with no records."""
        df = self.df.head(0).copy()
        add_MgNo(df)

    def test_one(self):
        """Check the ratio addition for one record."""
        df = self.df.head(1).copy()
        add_MgNo(df)

    def test_multiple(self):
        """Check the ratio addition for multiple records."""
        df = self.df.copy()
        add_MgNo(df)

    def test_weight_oxides(self):
        """Check accuracy of weight oxide data."""
        pass

    def test_molecular_oxides(self):
        """Check accuracy of molecular oxide data."""
        pass

    def test_weight_elemental(self):
        """Check accuracy of weight elemental data."""
        pass

    def test_molecular_elemental(self):
        """Check accuracy of molecular elemental data."""
        pass

    def test_Fe_components(self):
        """Check that the function works for multiple component Fe."""
        pass

class TestLambdaLnREE(unittest.TestCase):

    def setUp(self):
        self.rc = ReferenceCompositions()
        els = [i for i in REE() if not str(i)=='Pm']
        vals = [self.rc['Chondrite_PON'][el] for el in els]
        self.df = pd.DataFrame({k: v for (k, v) in zip(els, vals)}, index=[0])
        self.df.loc[1, :] = self.df.loc[0, :]
        self.default_degree = 5

    def test_exclude(self):
        """
        Tests the ability to generate lambdas from different element sets.
        """
        for exclude in [
                        [],
                        ['Pm'],
                        ['Pm', 'Eu'],
                        #['Pm', 'Eu', 'Ce']
                        ]:
            with self.subTest(exclude=exclude):
                ret = lambda_lnREE(self.df, exclude=exclude)
                self.assertTrue(ret.columns.size == self.default_degree)

    def test_degree(self):
        """
        Tests the ability to generate lambdas of different degree.
        """
        for degree in range(1, 6):
            with self.subTest(degree=degree):
                ret = lambda_lnREE(self.df, degree=degree)
                self.assertTrue(ret.columns.size == degree)

    def test_norm_to(self):
        """
        Tests the ability to generate lambdas using different normalisationsself."""
        for norm_to in self.rc.keys():
            data = self.rc[norm_to][self.df.columns]['value']
            if not pd.isnull(data).any():
                with self.subTest(norm_to=norm_to):
                    ret = lambda_lnREE(self.df, norm_to=norm_to)
                    self.assertTrue(ret.columns.size == self.default_degree)


if __name__ == '__main__':
    unittest.main()
