import unittest
from pyrolite.mineral.mineral import *
from pyrolite.mineral.db import *


class TestFormula2Elemental(unittest.TestCase):
    """Test the conversion of formula to elemental composition."""

    def setUp(self):
        self.text = "SiO2"
        self.formula = pt.formula(self.text)

    def test_formula_conversion(self):
        el = formula_to_elemental(self.text)

    def test_text_conversion(self):
        el = formula_to_elemental(self.formula)


class TestHeteromolecule(unittest.TestCase):
    """Test the molecule combination function for combining oxides."""

    def setUp(self):
        self.oxides = ["SiO2", "MgO", "CaO"]

    def test_generation(self):
        molecule = heteromolecule(self.oxides)


class TestMineralTemplate(unittest.TestCase):
    """Test the mineral template functionality."""

    def setUp(self):
        pass

    def test_repr(self):
        pass

    def test_str(self):
        pass


class TestMineral(unittest.TestCase):
    """Test the mineral functionality."""

    def setUp(self):
        self.ol = Mineral(
            "olivine",
            OLIVINE,
            pd.Series({"MgO": 42.06, "SiO2": 39.19, "FeO": 18.75}),
            endmembers={
                "Fo": "forsterite",
                "Fa": "fayalite",
                "Te": "tephroite",
                "Lie": "liebenbergite",
            },
        )

        self.pyx = Mineral(
            "pyroxene",
            PYROXENE,
            pd.Series(
                data=[57.10, 0.17, 0.70, 0.27, 0.60, 5.21, 0.17, 34.52, 0.62, 0.07],
                index="SiO2, TiO2, Al2O3, Cr2O3, Fe2O3, FeO, MnO, MgO, CaO, Na2O".split(
                    ", "
                ),
            ),
            endmembers={
                "En": "enstatite",
                "Fs": "ferrosilite",
                "Di": "diopside",
                "Hd": "hedenbergite",
                "Js": "johannsenite",
                "Es": "esseneite",
                "Jd": "jadeite",
                "Ae": "aegirine",
                "Ko": "kosmochlor",
                "Sm": "spodumene",
            },
        )

    def test_set_composition(self):
        """Check that different formulations of a composition can be accepted."""
        compositions = [
            pt.formula("Na2O Al2O3 SiO2"),
            {"Na2O": 1, "Al2O3": 1, "SiO2": 1},
        ]

        for comp in compositions:
            mineral = Mineral("mineral", composition=comp)

    def test_recalculate_cations(self):
        """
        Recalculation involves normalisation to a specific number of cations
        or oxygens.
        """
        for mineral in [self.pyx, self.ol]:
            recalc = mineral.recalculate_cations()

    def test_apfu(self):
        """
        Recalculation to provide atoms per formula units.
        """
        for mineral in [self.pyx, self.ol]:
            apfu = mineral.apfu()

    def test_endembmer_decompose(self):
        for mineral in [self.pyx, self.ol]:
            decomp = self.ol.endmember_decompose()
            assert np.isclose(sum([v for k, v in decomp.items()]), 1)

    def test_endmember_decompose_det_lim(self):
        """
        The endmember decomposition should return a decomposition with all
        endmembers which have abundances above a specified detection limit.
        """
        self.pyx.endmember_decompose(det_lim=0.0001)

    def test_endembmer_decompose_warning(self):
        """
        If endmbember decompose can't appropriately fit the mixture, it will return
        a warning. This is controlled by the sum of the cost function relative to
        the detection limit.
        """
        pass


if __name__ == "__main__":
    unittest.main()
