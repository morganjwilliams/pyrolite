import unittest
import numpy as np
from pyrolite.geochem.magma import *
from io import StringIO


class TestFeAt8MgO(unittest.TestCase):
    def setUp(self):
        self.FeOT = 2
        self.MgO = 4

    def test_default(self):
        feat8 = FeAt8MgO(self.FeOT, self.MgO)

    def test_close_to_8(self):
        feat8 = FeAt8MgO(self.FeOT, 8.0)
        self.assertTrue(np.isclose(feat8, self.FeOT, rtol=0.001))


class TestNaAt8MgO(unittest.TestCase):
    def setUp(self):
        self.Na2O = 2
        self.MgO = 4

    def test_default(self):
        naat8 = FeAt8MgO(self.Na2O, self.MgO)

    def test_close_to_8(self):
        naat8 = FeAt8MgO(self.Na2O, 8.0)
        self.assertTrue(np.isclose(naat8, self.Na2O, rtol=0.001))


class TestSCSS(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(
            StringIO(
                """	TempC	P(kb)	SiO2	FeO	TiO2	CaO	MgO	Al2O3	Fe2O3	Na2O	H2O
0	1384.96	1	49.3167	9.7718	0.9891	8.3127	15.3365	12.569	1.5261	1.2889	0
1	1374.96	1	49.4484	9.7798	1.0041	8.4345	14.8379	12.7592	1.5346	1.3084	0
2	1364.96	1	49.5791	9.7831	1.0189	8.5544	14.3503	12.9468	1.5427	1.3276	0
3	1344.96	1	49.8385	9.7761	1.0478	8.7896	13.4045	13.315	1.5578	1.3654	0
4	1324.96	1	50.095	9.7512	1.0761	9.0186	12.4976	13.6741	1.5715	1.4022	0
5	1304.96	1	50.3485	9.7087	1.1036	9.2414	11.6292	14.0242	1.5838	1.4381	0
6	1284.96	1	50.5993	9.6488	1.1305	9.4582	10.7989	14.3652	1.5946	1.4731	0
7	1264.96	1	50.8472	9.5718	1.1566	9.6691	10.0063	14.6974	1.6041	1.5071	0
8	1244.96	1	51.0924	9.4779	1.1821	9.8741	9.2508	15.0209	1.6122	1.5403	0
9	1224.96	1	51.3351	9.367	1.2069	10.0734	8.5317	15.336	1.6188	1.5726	0
10	1204.96	1	51.2184	9.3303	1.2455	10.3558	7.8186	15.7729	1.6422	1.6255	0
11	1184.96	1	51.2445	10.3252	1.5049	10.4316	6.9103	14.8799	1.8444	1.6694	0
12	1164.96	1	51.3461	11.8328	1.9663	10.1122	5.5988	13.7389	2.1171	1.7392	0
13	1144.96	1	51.489	13.0623	2.4044	9.4634	4.5369	13.0386	2.311	1.7966	0
14	1124.96	1	51.6163	14.0871	2.8312	8.9096	3.6373	12.3091	2.4754	1.8156	0
15	1104.96	1	51.7394	14.9315	3.253	8.4293	2.8643	11.7886	2.6168	1.805	0
"""
            ),
            sep="\t",
        )
        self.T = self.df["TempC"]
        self.P = self.df["P(kb)"]

    def test_default(self):
        T = self.T
        P = self.P
        sulfate, sulfide = SCSS(self.df, T=T, P=P)
        self.assertIsInstance(sulfide, np.ndarray)
        self.assertTrue((sulfide.ndim == 1) or (1 in sulfide.shape))
        self.assertTrue(np.nanmax(sulfide) < 1)

    def test_ppm_out(self):
        T = self.T
        P = self.P
        sulfate, sulfide = SCSS(self.df, T=T, P=P, outunit="ppm")
        self.assertTrue(np.nanmin(sulfide) > 1000)

    def test_kelvin(self):
        T = self.T + 273.15
        P = self.P
        sulfate, sulfide = SCSS(self.df, T=T, P=P, kelvin=False)
        self.assertIsInstance(sulfide, np.ndarray)

    def test_zeroD(self):
        T = self.T[0]
        P = self.P[0]
        sulfate, sulfide = SCSS(self.df.iloc[[0], :], T=T, P=P)
        self.assertIsInstance(sulfide, float)

    def test_geotherm(self):
        T = self.T
        P = self.P
        sulfate, sulfide = SCSS(self.df, T=T, P=P, grid="geotherm")
        self.assertIsInstance(sulfide, np.ndarray)
        self.assertTrue(sulfide.ndim == 2)

    def test_grid(self):
        T = self.T
        P = self.P
        sulfate, sulfide = SCSS(self.df, T=T, P=P, grid="grid")
        self.assertIsInstance(sulfide, np.ndarray)
        self.assertTrue(sulfide.ndim == 3)

    def test_grid_asymmetrical(self):
        T = self.T[:5]
        P = self.P[:3]
        sulfate, sulfide = SCSS(self.df, T=T, P=P, grid="grid")
        self.assertIsInstance(sulfide, np.ndarray)
        self.assertTrue(sulfide.ndim == 3)


if __name__ == "__main__":
    unittest.main()
