import unittest
import numpy as np
from pyrolite.util.synthetic import random_composition
from pyrolite.util.missing import md_pattern, cooccurence_pattern


class TestMDPattern(unittest.TestCase):
    def setUp(self):
        self.static = np.array(
            [[0, 1, 1], [0, np.nan, 1], [np.nan, 0, 1], [0, 1, np.nan], [1, np.nan, 0]]
        )
        self.rdata = random_composition(size=200, missing="MCAR")

    def test_default(self):
        pattern_ids, PD = md_pattern(self.static)
        self.assertTrue(pattern_ids.size == self.static.shape[0])
        self.assertTrue(len(PD.keys()) == 4)
        self.assertTrue(np.allclose([0, 1, 2, 3, 1], pattern_ids))
        self.assertTrue([i in PD for i in range(4)])

    def test_md_pattern_random(self):
        pattern_ids, PD = md_pattern(self.rdata)
        self.assertTrue(pattern_ids.size == self.rdata.shape[0])

        for i, d in PD.items():
            rows = pattern_ids == i
            where_present = np.arange(self.rdata.shape[1])[~d["pattern"]]
            where_not_present = np.arange(self.rdata.shape[1])[d["pattern"]]
            self.assertTrue([i not in where_not_present for i in where_present])
            self.assertTrue(np.isfinite(self.rdata[np.ix_(rows, where_present)]).all())
            self.assertTrue(
                (~np.isfinite(self.rdata[np.ix_(rows, where_not_present)])).all()
            )


class TestCooccurencePattern(unittest.TestCase):
    def setUp(self):
        self.static = np.array(
            [[0, 1, 1], [0, np.nan, 1], [np.nan, 0, 1], [0, 1, np.nan], [1, np.nan, 0]]
        )
        self.rdata = random_composition(size=200, missing="MCAR")

    def test_default(self):
        co_occur = cooccurence_pattern(self.static)
        self.assertTrue(co_occur.shape == (self.static.shape[1], self.static.shape[1]))

    def test_normalize(self):
        for normalize in [True, False]:
            with self.subTest(normalize=normalize):
                co_occur = cooccurence_pattern(self.static, normalize=normalize)
                self.assertTrue(
                    co_occur.shape == (self.static.shape[1], self.static.shape[1])
                )

    def test_log(self):
        for log in [True, False]:
            with self.subTest(log=log):
                co_occur = cooccurence_pattern(self.static, log=log)
                self.assertTrue(
                    co_occur.shape == (self.static.shape[1], self.static.shape[1])
                )

    def test_cooccurence_pattern_random(self):
        co_occur = cooccurence_pattern(self.rdata)
        self.assertTrue(co_occur.shape == (self.rdata.shape[1], self.rdata.shape[1]))


if __name__ == "__main__":
    unittest.main()
