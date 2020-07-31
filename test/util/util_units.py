import unittest
import pyrolite
from pyrolite.util.synthetic import normal_frame
from pyrolite.util.units import *
from pyrolite.util.units import scale, __UNITS__


class TestScaleFunction(unittest.TestCase):
    """Tests scale function generator."""

    def setUp(self):
        self.df = pd.DataFrame()
        self.df["units"] = pd.Series(list(__UNITS__.keys()))
        self.df["values"] = pd.Series(np.random.rand(self.df.index.size))

    def test_same_units(self):
        """Checks exchange between values with the same units is unity."""
        for to in __UNITS__.keys():
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
        fm_units = __UNITS__.keys()
        to_units = __UNITS__.keys()
        for to in to_units:
            for fm in [fu for fu in fm_units if not fu == to]:
                with self.subTest(fm=fm, to=to):
                    mult = scale(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not __UNITS__[to] == __UNITS__[fm]:
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
        to_units = __UNITS__.keys()
        for fm in fm_units:
            for to in to_units:
                with self.subTest(fm=fm, to=to):
                    mult = scale(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not __UNITS__[to] == __UNITS__[fm]:
                        self.assertFalse(
                            np.isclose(
                                self.df["values"].values * mult,
                                self.df["values"].values,
                            ).any()
                        )

    @unittest.expectedFailure
    def test_failure_on_unknown_unit_out(self):
        """Checks the function raises when unknown units are used for "to"."""
        fm_units = __UNITS__.keys()
        to_units = ["notaunit", "N/km2", "m/s", "ms-1"]
        for fm in fm_units:
            for to in to_units:
                with self.subTest(fm=fm, to=to):
                    mult = scale(fm, target_unit=to)
                    self.assertFalse(np.isnan(mult))
                    if not __UNITS__[to] == __UNITS__[fm]:
                        self.assertFalse(
                            np.isclose(
                                self.df["values"].values * mult,
                                self.df["values"].values,
                            ).any()
                        )


if __name__ == "__main__":
    unittest.main()
