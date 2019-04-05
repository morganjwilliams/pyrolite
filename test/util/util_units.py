import unittest
import pyrolite
from pyrolite.util.synthetic import test_df
from pyrolite.util.units import *


class TestScaleFunction(unittest.TestCase):
    """Tests scale function generator."""

    def setUp(self):
        self.df = pd.DataFrame()
        self.df["units"] = pd.Series(RELMASSS_UNITS.keys())
        self.df["values"] = pd.Series(np.random.rand(self.df.index.size))

    def test_same_units(self):
        """Checks exchange between values with the same units is unity."""
        for to in RELMASSS_UNITS.keys():
            with self.subTest(to=to):
                fm = to
                mult = scale(fm, target_unit=to)
                self.assertFalse(np.isnan(mult))
                self.assertTrue(
                    np.isclose(
                        self.df["values"].values * mult, self.df["values"].values
                    ).all()
                )

    def test_different_units(self):
        """Checks exchange between values with different units isn't unity."""
        fm_units = RELMASSS_UNITS.keys()
        to_units = RELMASSS_UNITS.keys()
        for to in to_units:
            for fm in [fu for fu in fm_units if not fu == to]:
                with self.subTest(fm=fm, to=to):
                    mult = scale(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not RELMASSS_UNITS[to] == RELMASSS_UNITS[fm]:
                        self.assertFalse(
                            np.isclose(
                                self.df["values"].values * mult,
                                self.df["values"].values,
                            ).any()
                        )

    @unittest.expectedFailure
    def test_failure_on_unknown_unit_in(self):
        """Checks the function raises when unknown units are used for "from"."""
        fm_units = ["notaunit", "N/km2", "m/s", "ms-1"]
        to_units = RELMASSS_UNITS.keys()
        for fm in fm_units:
            for to in to_units:
                with self.subTest(fm=fm, to=to):
                    mult = scale(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not RELMASSS_UNITS[to] == RELMASSS_UNITS[fm]:
                        self.assertFalse(
                            np.isclose(
                                self.df["values"].values * mult,
                                self.df["values"].values,
                            ).any()
                        )

    @unittest.expectedFailure
    def test_failure_on_unknown_unit_out(self):
        """Checks the function raises when unknown units are used for "to"."""
        fm_units = RELMASSS_UNITS.keys()
        to_units = ["notaunit", "N/km2", "m/s", "ms-1"]
        for fm in fm_units:
            for to in to_units:
                with self.subTest(fm=fm, to=to):
                    mult = scale(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not RELMASSS_UNITS[to] == RELMASSS_UNITS[fm]:
                        self.assertFalse(
                            np.isclose(
                                self.df["values"].values * mult,
                                self.df["values"].values,
                            ).any()
                        )


if __name__ == "__main__":
    unittest.main()
