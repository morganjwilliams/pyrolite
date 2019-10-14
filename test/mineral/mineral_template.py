import unittest
from pyrolite.mineral.template import *
from pyrolite.mineral.sites import *


class TestMineralTemplate(unittest.TestCase):
    """Test the mineral template functionality."""

    def setUp(self):
        self.min = MineralTemplate("TestMin")

    def test_set_structure(self):
        for structure in [[MX(), MX(), TX(), *[OX()] * 4], ["M", "M", "T"]]:
            self.min.set_structure(*structure)
            self.assertTrue(self.min.structure is not None)

    def test_repr(self):
        mins = [self.min]
        mins.append(self.min.copy())
        mins[1].set_structure([MX()])
        for min in mins:
            out = repr(min)

    def test_str(self):
        mins = [self.min]
        mins.append(self.min.copy())
        mins[1].set_structure([MX()])
        for min in mins:
            out = str(min)

    def test_hash(self):
        mins = [self.min]
        mins.append(self.min.copy())
        mins[1].set_structure([MX()])
        for min in mins:
            out = hash(min)


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

    def test_set_endmembers(self):
        min = self.pyx
        for em in [
            ["forsterite", "fayalite"],
            ["enstatite", "ferrosilite"],
            [
                pt.formula("SiO2"),
                pt.formula("MgO"),
                ("iron oxide", pt.formula("Fe2O3")),
            ],
        ]:
            min.set_endmembers(em)
            self.assertTrue(min.endmembers != {})

    def test_set_composition(self):
        """Check that different formulations of a composition can be accepted."""

        for comp in [pt.formula("Na2O Al2O3 SiO2"), {"Na2O": 1, "Al2O3": 1, "SiO2": 1}]:
            mineral = Mineral("mineral", composition=comp)

    def test_set_template(self):
        """Test different methods for setting templates."""
        for template in [PYROXENE, OLIVINE, [MX(), MX(), TX()], None]:
            mineral = Mineral("mineral", template=template)

    def test_recalculate_cations(self):
        """
        Recalculation involves normalisation to a specific number of cations
        or oxygens.
        """
        for mineral in [self.pyx, self.ol]:
            recalc = mineral.recalculate_cations()

    def test_apfu(self):
        """
        Test recalculation to provide atoms per formula units.
        """
        for mineral in [self.pyx, self.ol]:
            apfu = mineral.apfu()

    def test_calculate_occupancy(self):
        """
        Test site occupancy calculations.
        """
        for mineral, composition in zip(
            [self.pyx, self.ol], [None, self.ol.composition]
        ):
            mineral.calculate_occupancy()

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

    def test_get_site_occupancy(self):
        out = self.pyx.get_site_occupancy()

    def test_repr(self):
        for mineral in [self.pyx, self.ol]:
            out = repr(mineral)

    def test_str(self):
        for mineral in [self.pyx, self.ol]:
            out = str(mineral)

    def test_hash(self):
        for mineral in [self.pyx, self.ol]:
            out = hash(mineral)


if __name__ == "__main__":
    unittest.main()
