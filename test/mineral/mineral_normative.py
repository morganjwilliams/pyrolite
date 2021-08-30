import unittest
import pandas as pd
import numpy as np
from pyrolite.mineral.normative import (
    unmix,
    endmember_decompose,
    CIPW_norm,
    _aggregate_components,
    _update_molecular_masses,
    LeMaitre_Fe_correction,
    LeMaitreOxRatio,
)
from pyrolite.util.synthetic import normal_frame


class TestUnmix(unittest.TestCase):
    def setUp(self):
        # olivine Fo80 example
        self.comp = np.array([[0.4206, 0.3919, 0.1875], [0.4206, 0.3919, 0.1875]])
        self.parts = np.array(
            [
                [0.57294068, 0.42705932, 0.0],
                [0.0, 0.29485884, 0.70514116],
                [0.0, 1.0, 0.0],
                [0.25116001, 0.74883999, 0.0],
            ]
        )

    def test_default(self):
        res = unmix(self.comp, self.parts)

    def test_regularization(self):
        for ord in [1, 2]:
            with self.subTest(ord=ord):
                s = unmix(self.comp, self.parts, ord=ord)

    def test_det_lim(self):
        for det_lim in [0.001, 0.1, 0.5]:
            with self.subTest(det_lim=det_lim):
                s = unmix(self.comp, self.parts, det_lim=det_lim)


class TestEndmemberDecompose(unittest.TestCase):
    def setUp(self):
        # olivine Fo80 example
        self.df = pd.DataFrame(
            {"MgO": [42.06, 42.06], "SiO2": [39.19, 39.19], "FeO": [18.75, 18.75]}
        )

    def test_default(self):
        res = endmember_decompose(self.df)

    def test_endmembers_mineral_group(self):
        res = endmember_decompose(self.df, endmembers="olivine")

    def test_endmembers_list_mineralnames(self):
        res = endmember_decompose(self.df, endmembers=["forsterite", "fayalite"])

    def test_endmembers_list_formulae(self):
        res = endmember_decompose(self.df, endmembers=["Mg2SiO4", "Fe2SiO4"])

    def test_molecular(self):
        for molecular in [True, False]:
            with self.subTest(molecular=molecular):
                s = endmember_decompose(self.df, molecular=molecular)


class Test_AggregateComponents(unittest.TestCase):
    def setUp(self):
        pass


class Test_UpdateMolecularMasses(unittest.TestCase):
    def setUp(self):
        pass


class TestLeMaitreFeCorrection(unittest.TestCase):
    """
    This function uses new updates to convert_chemistry to allow
    acceptance of paired arrays for iron speciation.
    """

    def setUp(self):
        self.df = normal_frame(columns=["SiO2", "Fe2O3", "FeO", "MnO"])
        self.handler = "pyrolite.mineral.normative"

    def test_default(self):
        output = LeMaitre_Fe_correction(self.df)
        self.assertIsInstance(output, pd.DataFrame)
        self.assertTrue(all(output.columns == ["FeO", "Fe2O3"]))
        # from a synthetic dataset, no data should be missing or zero
        self.assertTrue(
            (~np.isfinite(output.values) | np.isclose(output.values, 0)).sum() == 0
        )

    def test_modes(self):
        pass

    def test_from_FeO(self):
        pass

    def test_from_FeOT(self):
        pass


class TestLeMatireOxRatio(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame(
            columns=["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "Na2O", "K2O"]
        )
        self.handler = "pyrolite.mineral.normative"

    def test_default(self):
        # should default to volcanic
        ratio = LeMaitreOxRatio(self.df)
        self.assertTrue(isinstance(ratio, pd.Series))
        self.assertEqual(ratio.name, "FeO/(FeO+Fe2O3)")
        self.assertTrue(
            np.isclose(ratio, LeMaitreOxRatio(self.df, mode="volcanic")).all()
        )

    def test_modes(self):
        """
        Check that the two modes work as expected.
        """
        valid_modes = ["Volcanic", "Plutonic"]
        # string should be normalised for comparison
        # only the start of the string is required
        valid_modes += (
            [m.lower() for m in valid_modes]
            + [m.upper() for m in valid_modes]
            + [m[:4] for m in valid_modes]
        )
        valid_modes += [None]  # will default to Volcanic
        for mode in valid_modes:
            with self.subTest(mode=mode):
                df = self.df
                with self.assertLogs(self.handler, level="DEBUG") as cm:
                    ratio = LeMaitreOxRatio(df, mode=mode)

    def test_missing_columns(self):
        """
        The function should warn where required columns are missing.

        For silica this is expected to fail; for the other components if they're
        missing the difference might be less infuencial so for the moment it will
        use reindexing and pass.
        """
        for missing in [["K2O"], ["K2O", "Na2O"]]:
            with self.subTest(missing=missing):
                df = self.df.drop(columns=missing)
                with self.assertLogs(self.handler, level="WARNING") as cm:
                    ratio = LeMaitreOxRatio(df)


class TestCIPW(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame(
            columns=["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO"]
            + ["MgO", "CaO", "Na2O", "K2O", "P2O5", "CO2", "SO3"]
        )
        self.handler = "pyrolite.mineral.normative"

    def test_default(self):
        norm = CIPW_norm(self.df)

    def noncritical_missing(self):
        # should logger.debug mentioning those missing
        for drop in (["CO2"], ["CO2", "SO3"]):
            with self.subTest(drop=drop):
                with self.assertLogs(self.handler, level="DEBUG") as cm:
                    norm = CIPW_norm(self.df.drop(columns=drop))
                    logging_output = cm.output

    def critical_missing(self):
        # should logger.warning mentioning the critical ones missing
        for drop in (["SiO2"], ["TiO2", "SO3"], ["SiO2", "MgO"]):
            with self.subTest(drop=drop):
                with self.assertLogs(self.handler, level="WARNING") as cm:
                    norm = CIPW_norm(self.df.drop(columns=drop))
                    logging_output = cm.output

    def test_adjust_all_fe(self):
        pass

    def test_fe_correction(self):
        pass

    def test_fe_correction_mode(self):
        pass
