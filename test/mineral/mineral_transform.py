import unittest
from pyrolite.mineral.transform import *


class TestFormula2Elemental(unittest.TestCase):
    """Test the conversion of formula to elemental composition."""

    def setUp(self):
        self.text = "SiO2"
        self.formula = pt.formula(self.text)

    def test_formula_conversion(self):
        el = formula_to_elemental(self.formula)

    def test_text_conversion(self):
        el = formula_to_elemental(self.text)

    def test_molecular_conversion(self):
        el = formula_to_elemental(self.formula, weight=False)


class TestMergeFormulae(unittest.TestCase):
    """Test the molecule combination function for combining oxides."""

    def setUp(self):
        self.oxides = ["SiO2", "MgO", "CaO"]

    def test_generation(self):
        molecule = merge_formulae(self.oxides)


class TestRecalcCations(unittest.TestCase):
    """Tests the standalone recalc_cations function."""

    def setUp(self):
        self.ol = pd.Series({"MgO": 42.06, "SiO2": 39.19, "FeO": 18.75})
        self.pyx = pd.Series(
            data=[57.10, 0.17, 0.70, 0.27, 0.60, 5.21, 0.17, 34.52, 0.62, 0.07],
            index="SiO2, TiO2, Al2O3, Cr2O3, Fe2O3, FeO, MnO, MgO, CaO, Na2O".split(
                ", "
            ),
        )

    def test_default(self):
        for min in [self.pyx, self.ol]:
            out = recalc_cations(min)


if __name__ == "__main__":
    unittest.main()
