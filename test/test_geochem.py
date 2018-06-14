import unittest
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

    def test_weightmolar_reversal(self, df, components):
        """
        Tests reversability of the wt-mol conversions.
        Examines differences between dataframes, and
        asserts that any discrepency is explained by np.nan components
        (and hence not actual differences).
        """
        wt_testdf = to_weight(to_molecular(df.loc[:, components]))
        self.assertTrue(np.isnan(
                        to_weight(
                        to_molecular(df.loc[:, components])
                        ).as_matrix()[~np.isclose(wt_testdf.as_matrix(),
                                      df.loc[:, components].as_matrix())
                                     ]
                       ).all())


class TestGetCations(unittest.TestCase):
    """Tests the cation calculator."""

    def test_none(self):
        """Check the function works for no cations."""
        pass

    def test_single(self):
        """Check the function works for a single cation."""
        pass

    def test_multiple(self):
        """Check the function works for multiple cations."""
        pass


class TestCommonElements(unittest.TestCase):
    """Tests the common element generator."""

    def test_cutoff(self):
        """Check the function works normal cutoff Z numbers."""
        pass

    def test_high_cutoff(self):
        """Check the function works silly high cutoff Z numbers."""
        pass

    def test_formula_output(self):
        """Check the function produces formula output."""
        pass

    def test_string_output(self):
        """Check the function produces string output."""
        pass


class TestREEElements(unittest.TestCase):
    """Tests the REE element generator."""

    def test_complete(self):
        """Check all REE are present."""
        pass

    def test_precise(self):
        """Check that only the REE are returned."""
        pass

    def test_formula_output(self):
        """Check the function produces formula output."""
        pass

    def test_string_output(self):
        """Check the function produces string output."""
        pass


class TestSimpleOxides(unittest.TestCase):
    """Tests the simple oxide generator."""

    def test_none(self):
        """Check the function returns no oxides for no elements in."""
        pass

    def test_one(self):
        """Check the function returns oxides for one element in."""
        pass

    @unittest.expectedFailure
    def test_multiple(self):
        """Check the function raises for muliple elements in."""
        pass


class TestCommonOxides(unittest.TestCase):
    """Tests the common oxide generator."""

    def test_none(self):
        """Check the function returns no oxides for no elements in."""
        pass

    def test_one(self):
        """Check the function returns oxides for one element in."""
        pass

    def test_multiple(self):
        """Check the function returns oxides for muliple elements in."""
        pass

    def test_formula_output(self):
        """Check the function produces formula output."""
        pass

    def test_string_output(self):
        """Check the function produces string output."""
        pass

    def test_addition(self):
        """Checks the addition functionality."""
        pass

    # As stands, unless addition == [], extras will be returned
    def test_precise(self):
        """Check that only relevant oxides are returned."""
        pass


class TestDevolatilise(unittest.TestCase):
    """Tests the devolatilisation transformation."""

    def test_none(self):
        """Check the function copes with no records."""
        pass

    def test_one(self):
        """Check the transformation functions for one record."""
        pass

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        pass

    def test_renorm(self):
        """Checks closure is achieved when renorm is used."""
        pass

    def test_exclude_precise(self):
        """Checks that exclusion occurrs correctly."""
        # There should be all those which weren't excluded

        # There should be no things which where included

        # There should be nothing else
        pass


class TestOxideConversion(unittest.TestCase):
    """Tests the pandas oxide conversion function generator."""

    def test_function_generation(self):
        """Check that a vaild function is returned."""
        pass

    def test_function_docstring(self):
        """Check the function docstring includes the oxide info."""
        pass

    def test_same(self):
        """Check the function retains unit for the same in-out."""
        pass

    def test_oxidise(self):
        """Check the function works for oxidation."""
        pass

    def test_reduce(self):
        """Check the function works for reduction."""
        pass

    def test_molecular(self):
        """Check that the generated function can convert molecular data."""
        pass


class TestRecalculateRedox(unittest.TestCase):
    """Tests the pandas dataframe redox conversion."""

    def test_none(self):
        """Check the function copes with no records."""
        pass

    def test_one(self):
        """Check the transformation functions for one record."""
        pass

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        pass

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
