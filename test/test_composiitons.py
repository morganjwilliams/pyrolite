import unittest
from pyrolite.compositions import *

class TestClose(unittest.TestCase):
    """Tests array closure operator."""

    def test_closure(self):
        """Checks that the closure operator works."""
        pass

    def test_single(self):
        """Checks results on single records."""
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass


class TestCompositionalMean(unittest.TestCase):
    """Tests pandas compositional mean operator."""

    def test_single(self):
        """Checks results on single records."""
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass

    @unittest.expectedFailure
    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should fail for this function
        pass

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass

    def test_closure(self):
        """Checks whether closure is preserved."""
        pass


class TestNANWeightedCompositionalMean(unittest.TestCase):
    """Tests pandas weighted compositonal NaN-mean operator."""

    def test_single(self):
        """Checks results on single records."""
        # Should not change result, once closure is considered
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for this function
        pass

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass

    def test_closure(self):
        """Checks whether closure is preserved."""
        pass


class TestNANWeightedMean(unittest.TestCase):
    """Tests pandas weighted NaN-mean operator."""

    def test_single(self):
        """Checks results on single records."""
        # Should not change result, once closure is considered
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for this function
        pass

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass

    def test_closure(self):
        """Checks whether closure is preserved."""
        pass


class TestCrossRatios(unittest.TestCase):
    """Tests pandas cross ratios utility."""

    def test_single(self):
        """Checks results on single records."""
        # Should return a single dxd array
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        # Should return an n x d x d array
        pass

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for this function
        pass

    def test_mean(self):
        """Checks whether the mean is accurate."""
        pass

class TestStandardiseAggregate(unittest.TestCase):
    """Tests pandas internal standardisation aggregation method."""

    def test_single(self):
        """Checks results on single records."""
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass

    def test_contrasting(self):
        """Checks results on multiple contrasting records."""
        # This should succeed for records which differ by all-but-one element
        pass

    def test_closure(self):
        """Checks whether closure is preserved when renormalisation is used."""
        pass

    def test_internal_standards(self):
        """Checks multiple internal standards work."""
        pass

    def test_fixed_record(self):
        """Checks whether assignment of fixed records works."""


class TestComplexStandardiseAggregate(unittest.TestCase):
    """Tests pandas complex internal standardisation aggregation method."""

    def test_single(self):
        """Checks results on single records."""
        pass

    def test_multiple(self):
        """Checks results on multiple records."""
        pass

    def test_complex(self):
        """Checks results on multiple contrasting records."""
        # This currently needs better imputation methods.
        pass

    def test_simple(self):
        """Check fallback to standardise aggregate."""
        pass

    def test_closure(self):
        """Checks whether closure is preserved when renormalisation is used."""
        pass

    def test_fixed_internal_standards(self):
        """Checks multiple internal standards work."""
        pass

    def test_fixed_record(self):
        """Checks whether assignment of fixed records works."""


class TestNaNCov(unittest.TestCase):
    """Tests the numpy nan covariance matrix utility."""

    def test_simple(self):
        """Checks whether non-nan covariances are correct."""
        pass

    def test_replace_method(self):
        """Checks whether the replacement method works."""
        pass

    def test_rowexclude_method(self):
        """Checks whether the traditional row-exclude method works."""
        pass

    def test_one_column_partial_nan(self):
        """Checks whether a single column containing NaN is processed."""
        pass

    def test_all_column_partial_nan(self):
        """Checks whether all columns containing NaNs is processed."""
        pass

    def test_one_column_all_nan(self):
        """Checks whether a single column all-NaN is processed."""
        pass

    def test_all_column_all_nan(self):
        """Checks whether all columns all-NaNs is processed."""
        pass


class TestRenormalise(unittest.TestCase):
    """Tests the pandas renormalise utility."""

    def test_closure(self):
        """Checks whether closure is achieved."""
        pass

    def test_components_selection(self):
        """Checks partial closure for different sets of components."""
        pass


class TestALR(unittest.TestCase):
    """Test the additive log ratio transformation."""

    def test_single(self):
        """Checks whether the function works on a single record."""
        pass

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        pass

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        pass

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        pass


class TestCLR(unittest.TestCase):
    """Test the centred log ratio transformation."""

    def test_single(self):
        """Checks whether the function works on a single record."""
        pass

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        pass

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        pass

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        pass


class TestOrthagonalBasis(unittest.TestCase):
    """Test the orthagonal basis generator for ILR transformation."""

    def test_orthagonal_basis(self):
        """Checks orthagonality of the transformation basis."""
        pass


class TestILR(unittest.TestCase):
    """Test the isometric log ratio transformation."""

    def test_single(self):
        """Checks whether the function works on a single record."""
        pass

    def test_multiple(self):
        """Checks whether the function works on multiple records."""
        pass

    def test_isomorphism_single(self):
        """Checks that the function is reversible for a record."""
        pass

    def test_isomorphism_multiple(self):
        """Checks that the function is reversible for multiple records."""
        pass


class TestLogTransformers(unittest.TestCase):
    """Checks the scikit-learn transformer classes."""

    def test_ALR_transformer(self):
        """Test the isometric log ratio transfomer."""
        pass

    def test_CLR_transformer(self):
        """Test the isometric log ratio transfomer."""
        pass

    def test_ILR_transformer(self):
        """Test the isometric log ratio transfomer."""
        pass


if __name__ == '__main__':
    unittest.main()
