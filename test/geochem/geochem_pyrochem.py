import unittest
import numpy as np
import pandas as pd
import pyrolite.geochem
from pyrolite.comp.codata import renormalise
from pyrolite.util.synthetic import normal_frame, normal_series
from pyrolite.geochem.norm import get_reference_composition

# [print("# " + i) for i in dir(df.pyrochem) if "__" not in i and not i.startswith("_")]


class TestPyrochem(unittest.TestCase):
    """
    Test the pyrochem dataframe accessor interface to
    pyrolite.geochem functions.
    """

    def setUp(self):
        cols = [
            "MgO",
            "SiO2",
            "CaO",
            "FeO",
            "Ti",
            "Hf",
            "Zr",
            "H2O",
            "Sr87_Sr86",
            "87Sr/86Sr",
            "87Sr/86Sri",
        ] + pyrolite.geochem.REE()
        self.df = normal_frame(size=4, columns=cols)
        self.df = renormalise(self.df)

    # pyrolite.geochem.ind functions

    def test_pyrochem_indexes(self):
        obj = self.df
        for index in [
            "list_elements",
            "list_oxides",
            "list_REE",
            "list_isotope_ratios",
        ]:
            with self.subTest(index=index):
                out = getattr(obj.pyrochem, index)
                self.assertIsInstance(out, list)

    def test_pyrochem_subsetters(self):
        obj = self.df
        for subset in [
            "REE",
            "REY",
            "elements",
            "oxides",
            "isotope_ratios",
            "compositional",
        ]:
            with self.subTest(subset=subset):
                out = getattr(obj.pyrochem, subset)
                self.assertIsInstance(out, obj.__class__)  # in this case a dataframe

    def test_pyrochem_subsetter_assignment(self):
        obj = self.df
        for subset in [
            "REE",
            "REY",
            "elements",
            "oxides",
            "isotope_ratios",
            "compositional",
        ]:
            with self.subTest(subset=subset):
                setattr(obj.pyrochem, subset, getattr(obj.pyrochem, subset) * 1.0)

    # pyrolite.geochem.parse functions

    def test_pyrochem_check_multiple_cation_inclusion(self):
        obj = self.df.copy(deep=True)
        cations = obj.pyrochem.check_multiple_cation_inclusion()
        self.assertTrue(len(cations) == 0)

    def test_pyochem_parse_chem(self):
        obj = self.df.copy(deep=True)
        start_cols = obj.columns
        out = obj.pyrochem.parse_chem()
        self.assertTrue(len(out.columns) == len(start_cols))
        self.assertTrue(
            all([a == b for (a, b) in zip(out.columns, start_cols) if "/" not in a])
        )

    # pyrolite.geochem.transform functions

    def test_pyrochem_to_molecular(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        start = obj.values
        out = obj.pyrochem.to_molecular()
        self.assertFalse(np.isclose(out.values.flatten(), start.flatten()).any())

    def test_pyrochem_to_weight(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        start = obj.values
        out = obj.pyrochem.to_weight()
        self.assertFalse(np.isclose(out.values.flatten(), start.flatten()).any())

    def test_pyrochem_add_MgNo(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        obj.pyrochem.add_MgNo()
        self.assertIn("Mg#", obj.columns)

    def test_pyrochem_add_MgNo_ferricferrous(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        obj.pyrochem.add_MgNo()
        obj.pyrochem.recalculate_Fe(to=dict(FeO=0.9, Fe2O3=0.1))
        obj.pyrochem.add_MgNo(name="Mg#2")
        self.assertIn("Mg#", obj.columns)
        self.assertIn("Fe2O3", obj.columns)
        self.assertIn("Mg#2", obj.columns)
        self.assertFalse(np.isclose(obj["Mg#"].values, obj["Mg#2"].values).any())

    def test_pyrochem_add_ratio(self):
        obj = self.df.copy(deep=True)

        for ratio in ["MgO/SiO2", "MgO/Ti", "Lu/Hf", "Mg/TiO2"]:
            with self.subTest(ratio=ratio):
                obj.pyrochem.add_ratio(ratio)
                self.assertIn(ratio, obj.columns)

    def test_pyrochem_aggregate_element(self):
        obj = self.df.copy(deep=True)
        target = "Fe"
        out = obj.pyrochem.aggregate_element(target)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertTrue(target in out.columns)

    def test_pyrochem_devolatilise(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.devolatilise()
        self.assertNotIn("H2O", out.columns)

    def test_pyrochem_elemental_sum(self):
        obj = self.df.copy(deep=True)
        Mg = obj.pyrochem.elemental_sum("Mg")
        self.assertFalse(np.isclose(Mg, obj.MgO).any())

    def test_pyrochem_lambda_lnREE(self):
        obj = self.df.copy(deep=True)
        lambdas = obj.pyrochem.lambda_lnREE()
        self.assertIsInstance(lambdas, pd.DataFrame)
        self.assertIn("λ0", lambdas.columns)

    def test_pyrochem_recalculate_Fe(self):
        obj = self.df.copy(deep=True)
        obj.pyrochem.recalculate_Fe(to=dict(FeO=0.9, Fe2O3=0.1))
        self.assertIn("Fe2O3", obj.columns)
        self.assertIn("FeO", obj.columns)
        self.assertTrue((obj.FeO.values > obj.Fe2O3.values).all())

    def test_pyrochem_convert_chemistry(self):
        obj = self.df.copy(deep=True)
        obj = obj.pyrochem.convert_chemistry(
            to=["MgO", "Si", "Ti", "HfO2", "La2O3", dict(FeO=0.9, Fe2O3=0.1)]
        )
        self.assertIn("Fe2O3", obj.columns)
        self.assertIn("Si", obj.columns)
        self.assertIn("Si", obj.columns)
        self.assertTrue((obj.FeO.values > obj.Fe2O3.values).all())

    # pyrolite.geochem.norm functions

    def test_pyrochem_normalize_to_str(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.normalize_to("Chondrite_PON")

    def test_pyrochem_normalize_to_composition(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.normalize_to(get_reference_composition("Chondrite_PON"))

    def test_pyrochem_normalize_to_array(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.normalize_to(np.ones(obj.columns.size))

    def test_pyrochem_denormalize_from_str(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.denormalize_from("Chondrite_PON")

    def test_pyrochem_denormalize_from_composition(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.denormalize_from(get_reference_composition("Chondrite_PON"))

    def test_pyrochem_denormalize_from_array(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        out = obj.pyrochem.denormalize_from(np.ones(obj.columns.size))

    def test_pyrochem_scale(self):
        obj = self.df.copy(deep=True).pyrochem.compositional
        REEppm = obj.pyrochem.REE.pyrochem.scale("wt%", "ppm")
        self.assertFalse(np.isclose(REEppm.values, obj.pyrochem.REE.values).any())
        self.assertTrue((REEppm.values > obj.pyrochem.REE.values).all())
