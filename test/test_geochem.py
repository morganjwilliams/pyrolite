import unittest
import pandas as pd
import numpy as np
from pyrolite.geochem import *

class TestToMolecular(unittest.TestCase):
    """Tests pandas molecular conversion operator."""

    def test_closure(self):
        """Checks that the closure operator works."""
        pass

    def test_single(self):
        """Checks results on single records."""
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass


class TestToWeight(unittest.TestCase):
    """Tests pandas weight conversion operator."""

    def test_closure(self):
        """Checks that the closure operator works."""
        pass

    def test_single(self):
        """Checks results on single records."""
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass


class TestWeightMolarReversal(unittest.TestCase):
    """Tests the reversability of weight-molar unit transformations."""

    def setUp(self):
        self.df = pd.DataFrame({'MgO':20.0, 'SiO2':30.0, 'K2O':5.0, 'Na2O':2.0},
                               index=[0])
        self.components = ['MgO', 'SiO2', 'K2O']

    def test_weightmolar_reversal_renormFalse(self):
        """
        Tests reversability of the wt-mol conversions.
        Examines differences between dataframes, and
        asserts that any discrepency is explained by np.nan components
        (and hence not actual differences).
        """
        wt_testdf = to_weight(to_molecular(self.df.loc[:, self.components],
                                          renorm=False),
                              renorm=False)
        # Where values are not close, it's because of nans
        whereclose = np.isclose(wt_testdf.values,
                                self.df.loc[:, self.components].values)
        self.assertTrue(np.isnan(wt_testdf.values[~whereclose]).all())

    def test_weightmolar_reversal_renormTrue(self):
        """
        Tests reversability of the wt-mol conversions.
        Examines differences between dataframes, and
        asserts that any discrepency is explained by np.nan components
        (and hence not actual differences).
        """
        wt_testdf = to_weight(to_molecular(self.df.loc[:, self.components],
                                          renorm=True),
                              renorm=True)
        # Where values are not close, it's because of nans
        whereclose = np.isclose(wt_testdf.values,
                                renormalise(self.df.loc[:,
                                            self.components]).values)
        self.assertTrue(np.isnan(wt_testdf.values[~whereclose]).all())

    def test_molarweight_reversal_renormTrue(self):
        """
        Tests reversability of the mol-wt conversions.
        Examines differences between dataframes, and
        asserts that any discrepency is explained by np.nan components
        (and hence not actual differences).
        """
        mol_testdf = to_molecular(to_weight(self.df.loc[:, self.components],
                                          renorm=True),
                              renorm=True)
        # Where values are not close, it's because of nans
        whereclose = np.isclose(mol_testdf.values,
                                renormalise(self.df.loc[:,
                                            self.components]).values)
        self.assertTrue(np.isnan(mol_testdf.values[~whereclose]).all())

    def test_molarweight_reversal_renormFalse(self):
        """
        Tests reversability of the mol-wt conversions.
        Examines differences between dataframes, and
        asserts that any discrepency is explained by np.nan components
        (and hence not actual differences).
        """
        mol_testdf = to_molecular(to_weight(self.df.loc[:, self.components],
                                          renorm=False),
                              renorm=False)
        # Where values are not close, it's because of nans
        whereclose = np.isclose(mol_testdf.values,
                                self.df.loc[:,self.components].values)
        self.assertTrue(np.isnan(mol_testdf.values[~whereclose]).all())


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
                self.assertTrue(common_elements(cutoff=cutoff)[-1].number==cutoff)

    def test_high_cutoff(self):
        """Check the function works silly high cutoff Z numbers."""
        for cutoff in [119, 1000, 10000]:
            with self.subTest(cutoff=cutoff):
                self.assertTrue(len(common_elements(cutoff=cutoff))<130)
                self.assertTrue(common_elements(cutoff=cutoff)[-1].number<cutoff)

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


class TestREEElements(unittest.TestCase):
    """Tests the REE element generator."""

    def setUp(self):
        self.min_z = 57
        self.max_z = 71

    def test_complete(self):
        """Check all REE are present."""
        REE = REE_elements(output='formula')
        ns = [el.number for el in REE]
        for n in range(self.min_z, self.max_z + 1):
            with self.subTest(n=n):
                self.assertTrue(n in ns)

    def test_precise(self):
        """Check that only the REE are returned."""
        REE = REE_elements(output='formula')
        ns = [el.number for el in REE]
        self.assertTrue(min(ns) == self.min_z)
        self.assertTrue(max(ns) == self.max_z)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for el in REE_elements(output='formula'):
            with self.subTest(el=el):
                self.assertIs(type(el), type(pt.elements[0]))

    def test_string_output(self):
        """Check the function produces string output."""
        for el in REE_elements(output='string'):
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
        df = pd.DataFrame(data=self.one_row)
        df.columns =  self.cols
        for renorm in [True, False]:
            with self.subTest(renorm=renorm):
                devdf = devolatilise(df, renorm=renorm)
                equality = (devdf.values == \
                            df.loc[:, devdf.columns].values).all()
                # For renorm = True, values will not be the same
                # For renorm = False, values should be the same
                self.assertTrue(equality != renorm)

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
                print(doc)
                self.assertTrue((str(oxin) in doc) and (str(oxin) in doc))
                self.assertTrue(f'{oxin} to {oxout}' in doc)

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
        self.one_row = np.array([[0.5, 0.3, 0.2]])
        self.two_rows = np.array([[0.5, 0.3, 0.2],
                                  [0.5, 0.3, 0.2]])

    def test_none(self):
        """Check the function copes with no records."""
        df = pd.DataFrame(columns=self.cols )
        self.assertTrue(recalculate_redox(df) is not None)
        self.assertIs(type(recalculate_redox(df)), pd.DataFrame)
        self.assertEqual(recalculate_redox(df).index.size, 0)

    def test_one(self):
        """Check the transformation functions for one record."""
        df = pd.DataFrame(self.one_row, columns=self.cols)
        self.assertEqual(recalculate_redox(df).index.size,
                         self.one_row.shape[0])

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        df = pd.DataFrame(self.two_rows, columns=self.cols)
        self.assertEqual(recalculate_redox(df).index.size,
                         self.two_rows.shape[0])

    def test_to_oxidised(self):
        """Check the oxidised form is returned when called."""
        pass

    def test_to_reduced(self):
        """Check the reduced form is returned when called."""
        pass

    def test_renorm(self):
        """Checks closure is achieved when renorm is used."""
        pass

    def test_total_suffix(self):
        """Checks that different suffixes can be used."""
        pass

    def test_columns_dropped(self):
        """Checks that only one redox state is found in output."""
        pass

class TestAggregateCation(unittest.TestCase):
    """Tests the pandas dataframe cation aggregation transformation."""

    def test_none(self):
        """Check the transformation copes with no records."""
        pass

    def test_one(self):
        """Check the transformation functions for one record."""
        pass

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        pass

    def test_different_cations(self):
        """Checks multiple cations can be accomdated."""

        # use subtests per cation
        pass

    def test_oxide_return(self):
        """Checks that oxide forms are returned."""

        # Check presence

        # Check absence of others

        # Check preciseness

        # Check no additional features added

        pass

    def test_element_return(self):
        """Checks that element forms are returned."""

        # Check presence

        # Check absence of others

        # Check preciseness

        # Check no additional features added

        pass

    def check_unit_scale(self):
        """Checks that the unit scales are used."""


class TestMultipleCationInclusion(unittest.TestCase):
    """Tests the pandas dataframe multiple inclusion checking."""

    def test_none(self):
        """Check the function copes with no records."""
        pass

    def test_one(self):
        """Check the transformation functions for one record."""
        pass

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        pass

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
        pass

    def test_alias(self):
        """Check that aliases can be used."""
        pass

    def test_convert(self):
        """Check that lambda conversion works."""
        pass


class TestAddMgNo(unittest.TestCase):
    """Tests the MgNo addition."""

    def test_none(self):
        """Check the ratio addition copes with no records."""
        pass

    def test_one(self):
        """Check the ratio addition for one record."""
        pass

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

class TestSpiderplot(unittest.TestCase):
    """Tests the Spiderplot functionality."""

    def test_none(self):
        """Test generation of plot with no data."""
        pass

    def test_one(self):
        """Test generation of plot with one record."""
        pass

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        pass

    def test_no_axis_specified(self):
        """Test generation of plot without axis specified."""
        pass

    def test_axis_specified(self):
        """Test generation of plot with axis specified."""
        pass

    def test_no_components_specified(self):
        """Test generation of plot with no components specified."""
        pass

    def test_components_specified(self):
        """Test generation of plot with components specified."""
        pass

    def test_plot_off(self):
        """Test plot generation with plot off."""
        pass

    def test_fill(self):
        """Test fill functionality is available."""
        pass

    def test_valid_style(self):
        """Test valid styling options."""
        pass

    def test_irrellevant_style_options(self):
        """Test stability under additional kwargs."""
        pass

    @unittest.expectedFailure
    def test_invalid_style_options(self):
        """Test stability under invalid style values."""
        pass

class TestTernaryplot(unittest.TestCase):
    """Tests the Ternaryplot functionality."""

    def test_none(self):
        """Test generation of plot with no data."""
        pass

    def test_one(self):
        """Test generation of plot with one record."""
        pass

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        pass

    def test_tax_returned(self):
        """Check that the axis item returned is a ternary axis."""
        pass

    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        pass


if __name__ == '__main__':
    unittest.main()
