import unittest
import numpy as np
import pyrolite.comp
from pyrolite.util.synthetic import normal_frame
from pyrolite.geochem.ind import REE

np.random.seed(81)


class TestPyroComp(unittest.TestCase):
    def setUp(self):
        self.cols = ["MgO", "SiO2", "CaO"]

        # can run into interesting singular matrix errors with bivariate random data
        self.tridf = normal_frame(columns=self.cols, size=100)
        self.bidf = self.tridf.loc[:, self.cols[:2]]
        self.multidf = normal_frame(columns=REE(), size=100)

    def test_renormalise_default(self):
        df = self.bidf.copy(deep=True) * 100  # copy df
        df["SiO2"] += 50  # modify df
        dfval = df["SiO2"].values[0]
        out = df.pyrocomp.renormalise()  # renorm
        self.assertTrue(df["SiO2"].values[0] == dfval)  # check original hasn't changed
        self.assertTrue((np.allclose(out.sum(axis=1), 100.0)))  # check output

    def test_renormalise_components(self):
        df = self.tridf.copy(deep=True) * 100  # copy df
        out = df.pyrocomp.renormalise(components=self.cols[:2])  # renorm
        self.assertTrue(
            (np.allclose(out[self.cols[:2]].sum(axis=1), 100.0))
        )  # check output

    def test_ALR_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        out = df.pyrocomp.ALR()
        self.assertTrue("ALR_index" in out.attrs)
        self.assertTrue("inverts_to" in out.attrs)
        self.assertTrue(out.attrs["inverts_to"] == self.cols)

    def test_ALR_name_index(self):
        df = self.tridf.copy(deep=True)  # copy df
        ind = "SiO2"
        out = df.pyrocomp.ALR(ind=ind)
        self.assertTrue("ALR_index" in out.attrs)
        self.assertTrue("inverts_to" in out.attrs)
        self.assertTrue(all([ind in colname for colname in out.columns]))
        self.assertTrue(out.attrs["inverts_to"] == self.cols)

    def test_inverse_ALR_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        intermediate = df.pyrocomp.ALR()
        out = intermediate.pyrocomp.inverse_ALR()
        self.assertTrue((out.columns == self.cols).all())
        self.assertTrue(np.allclose(out, df))

    def test_CLR_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        out = df.pyrocomp.CLR()
        self.assertTrue("inverts_to" in out.attrs)
        self.assertTrue(out.attrs["inverts_to"] == self.cols)

    def test_inverse_CLR_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        intermediate = df.pyrocomp.CLR()
        out = intermediate.pyrocomp.inverse_CLR()
        self.assertTrue((out.columns == self.cols).all())
        self.assertTrue(np.allclose(out, df))

    def test_ILR_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        out = df.pyrocomp.ILR()
        self.assertTrue("inverts_to" in out.attrs)
        self.assertTrue(out.attrs["inverts_to"] == self.cols)

    def test_inverse_ILR_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        intermediate = df.pyrocomp.ILR()
        out = intermediate.pyrocomp.inverse_ILR()
        self.assertTrue((out.columns == self.cols).all())
        self.assertTrue(np.allclose(out, df))

    def test_boxcox_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        out = df.pyrocomp.boxcox()
        self.assertTrue("boxcox_lmbda" in out.attrs)

    def test_inverse_boxcox_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        intermediate = df.pyrocomp.boxcox()
        out = intermediate.pyrocomp.inverse_boxcox()
        self.assertTrue((out.columns == self.cols).all())
        self.assertTrue(np.allclose(out, df))

    def test_logratiomean_default(self):
        df = self.tridf.copy(deep=True)  # copy df
        out = df.pyrocomp.logratiomean()

    def test_invert_transform(self):
        df = self.tridf.copy(deep=True)  # copy df
        for tfm in [df.pyrocomp.ALR, df.pyrocomp.CLR, df.pyrocomp.ILR]:
            with self.subTest(tfm=tfm):
                out = tfm()
                out_inv = out.pyrocomp.invert_transform()
                self.assertTrue(np.allclose(out_inv.values, df.values))

    def test_labelling(self):
        df = self.tridf.copy(deep=True)  # copy df
        # test that the label modes can be called
        for mode in ["numeric", "simple", "latex"]:
            for m in [df.pyrocomp.ALR, df.pyrocomp.CLR, df.pyrocomp.ILR]:
                with self.subTest(mode=mode, m=m):
                    out = m(label_mode=mode)

    def test_labelling_invalid(self):
        df = self.tridf.copy(deep=True)  # copy df
        # test that the label modes can be called
        for mode in ["math", "bogus"]:
            for m in [df.pyrocomp.ALR, df.pyrocomp.CLR, df.pyrocomp.ILR]:
                with self.subTest(mode=mode, m=m):
                    with self.assertRaises(NotImplementedError):
                        out = m(label_mode=mode)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
