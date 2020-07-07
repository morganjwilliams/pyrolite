import unittest
import numpy as np
import pandas as pd
from pyrolite.geochem.ind import REE
from pyrolite.geochem.parse import *


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


class TestMultipleCationInclusion(unittest.TestCase):
    """Tests the pandas dataframe multiple inclusion checking."""

    def setUp(self):
        self.cols = ["MgO", "FeO", "Fe2O3", "Mg", "Fe", "FeOT"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )

    def test_none(self):
        """Check the function copes with no records."""
        # Note that this function runs from columns - doesn't need records
        df = self.df.head(0)
        self.assertTrue(len(check_multiple_cation_inclusion(df)) > 0)
        self.assertTrue(
            all(
                [
                    i.__str__() in ["Mg", "Fe"]
                    for i in check_multiple_cation_inclusion(df)
                ]
            )
        )

    def test_one(self):
        """Check the transformation functions for one record."""
        df = self.df.head(1)
        self.assertTrue(len(check_multiple_cation_inclusion(df)) > 0)
        self.assertTrue(
            all(
                [
                    i.__str__() in ["Mg", "Fe"]
                    for i in check_multiple_cation_inclusion(df)
                ]
            )
        )

    def test_multiple(self):
        """Check the transformation functions for multiple records."""
        df = self.df
        self.assertTrue(len(check_multiple_cation_inclusion(df)) > 0)
        self.assertTrue(
            all(
                [
                    i.__str__() in ["Mg", "Fe"]
                    for i in check_multiple_cation_inclusion(df)
                ]
            )
        )

    def test_exclusion(self):
        """Checks that exclusions are properly handled."""
        # Check that excluded components aren't considered
        pass

    def test_output(self):
        """Checks that the list returned is complete."""

        # Check complete

        # Check precise

        pass


class TestReprIsotopeRatio(unittest.TestCase):
    def setUp(self):
        self.expect = "87Sr/86Sr"
        self.expect_succeed = [
            "87Sr/86Sr",
            "87Sr_86Sr",
            "Sr87/Sr86",
            "Sr87_Sr86",
            "87Sr 86Sr",
        ]
        self.expect_fail = ["87Sr/Sr86", "87Sr_Sr86", "87Sr Sr86"]

    def test_default(self):
        for ratio in self.expect_succeed:
            with self.subTest(ratio=ratio):
                out = repr_isotope_ratio(ratio)
                self.assertEqual(out, self.expect)

        for ratio in self.expect_fail:
            with self.subTest(ratio=ratio):
                out = repr_isotope_ratio(ratio)
                self.assertEqual(out, ratio)  # hasn't changed the string


if __name__ == "__main__":
    unittest.main()
