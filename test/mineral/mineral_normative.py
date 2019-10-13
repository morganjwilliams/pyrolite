import unittest
import pandas as pd
import numpy as np
from pyrolite.mineral.normative import unmix, endmember_decompose


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
