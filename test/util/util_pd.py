import os, time
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from pyrolite.util.synthetic import normal_frame, normal_series
from pyrolite.util.general import temp_path, remove_tempdir
from pyrolite.util.meta import subkwargs
from pyrolite.util.pd import *


class TestColumnOrderedAppend(unittest.TestCase):
    def setUp(self):
        pass

    def test_column_order(self):
        pass

    def test_index_preservation(self):
        pass


class TestReadTable(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path()
        self.fn = self.dir / "test_read_table.csv"
        self.files = [self.fn.with_suffix(s) for s in [".csv"]]  # ".xlsx",
        self.expect = pd.DataFrame(
            np.ones((2, 2)), columns=["C1", "C2"], index=["i0", "i1"]
        )
        for fn, ex in zip(self.files, ["to_csv"]):  # make some csvs # "to_excel",
            kw = dict()  # engine="openpyxl"
            getattr(self.expect, ex)(str(fn), **subkwargs(kw, getattr(self.expect, ex)))

    def test_read_csv(self):
        f = self.fn.with_suffix(".csv")
        df = read_table(f, index=0)
        self.assertTrue((df.values == self.expect.values).all())
        self.assertTrue((df.columns == np.array(["C1", "C2"])).all())



class TestAccumulate(unittest.TestCase):
    def setUp(self):
        self.df0 = normal_frame()
        self.others = [normal_frame()] * 4

    def test_column_order(self):
        result = accumulate([self.df0] + self.others)
        self.assertTrue(all(result.columns == self.df0.columns))

    def test_index_preservation(self):
        result = accumulate([self.df0] + self.others)
        # The range index should just be repeated 5 times, not reset
        self.assertTrue(
            all(
                [
                    res == exp
                    for (res, exp) in zip(
                        list(result.index.values),
                        list(np.tile(self.df0.index.values, 5)),
                    )
                ]
            )
        )


class TestToFrame(unittest.TestCase):
    """Test the 'to_frame' utility dataframe conversion function."""

    def setUp(self):
        self.ser = normal_series()
        self.df = normal_frame()

    def normal_frame_column_order(self):
        result = to_frame(self.df)
        self.assertTrue((result.columns == self.df.columns).all())

    def normal_series_column_order(self):
        result = to_frame(self.ser)
        self.assertTrue((result.columns == self.ser.index).all())

    def normal_frame_index_preservation(self):
        result = to_frame(self.df)
        self.assertTrue((result.index == self.df.index).all())

    def normal_seriesies_conversion(self):
        result = to_frame(self.ser)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_non_convertable(self):
        for noconv in [None, [0, 1, [1]]]:
            with self.subTest(noconv=noconv):
                with self.assertRaises(NotImplementedError) as cm:
                    result = to_frame(noconv)


class TestToSer(unittest.TestCase):
    def setUp(self):
        self.ser = normal_series()
        self.df = normal_frame()

    def test_single_column(self):
        result = to_ser(self.df.iloc[:, 0])
        self.assertTrue((result.index == self.df.index).all())

    def test_single_row(self):
        result = to_ser(self.df.iloc[0, :])
        self.assertTrue((result.index == self.df.columns).all())

    def test_assertion_error_mulitcolumn(self):
        with self.assertRaises(AssertionError) as cm:
            result = to_ser(self.df)

    def test_non_convertable(self):
        for noconv in [None, [0, 1, [1]]]:
            with self.subTest(noconv=noconv):
                with self.assertRaises(NotImplementedError) as cm:
                    result = to_ser(noconv)


class TestToNumeric(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame().applymap(str)

    def test_numeric(self):
        df = self.df
        result = to_numeric(df)
        self.assertTrue((result.dtypes == "float64").all())

    def test_error_methods(self):
        df = self.df
        df.loc[0, "SiO2"] = "Low"
        for method in ["ignore", "raise", "coerce"]:
            with self.subTest(method=method):
                try:
                    result = to_numeric(df, errors=method)
                    self.assertTrue(method in ["ignore", "coerce"])
                    if method == "ignore":
                        self.assertTrue(result.loc[0, "SiO2"] == "Low")
                    else:
                        self.assertTrue(pd.isnull(result.loc[0, "SiO2"]))
                except ValueError:  # should raise with can't parse 'low'
                    self.assertTrue(method == "raise")


class TestOutliers(unittest.TestCase):
    def setUp(self):
        self.df = normal_frame()

    def test_exclude(self):
        for exclude in [True, False]:
            with self.subTest(exclude=exclude):
                ret = outliers(self.df, exclude=exclude)
                self.assertEqual(ret.size > 0.5 * self.df.size, exclude)

    def test_quantile(self):
        for q in [(0.02, 0.98), (0.2, 0.8)]:
            with self.subTest(q=q):
                ret = outliers(self.df, quantile_select=q)

    def test_logquantile(self):
        for logquantile in [True, False]:
            with self.subTest(logquantile=logquantile):
                ret = outliers(self.df, logquantile=logquantile)


class TestConcatColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {0: ["a", "b", "c"], 1: ["d", "e", "f"]}, index=["A", "B", "C"]
        ).T

    def test_default(self):
        out = concat_columns(self.df)
        self.assertTrue((out == pd.Series(["abc", "def"])).all())

    def test_columns(self):
        out = concat_columns(self.df, columns=["A", "B"])
        self.assertTrue((out == pd.Series(["ab", "de"])).all())


class TestUniquesFromConcat(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {0: ["a", "b", "c"], 1: ["d", "e", "f"]}, index=["A", "B", "C"]
        ).T

    def test_default(self):
        out = uniques_from_concat(self.df)
        self.assertTrue(out.index.size == len(out.unique()))

    def test_columns(self):
        out = uniques_from_concat(self.df, columns=["A", "B"])
        self.assertTrue(out.index.size == len(out.unique()))

    def test_hashit(self):
        for h in [True, False]:
            with self.subTest(h=h):
                out = uniques_from_concat(self.df, hashit=h)
                self.assertTrue(out.index.size == len(out.unique()))
                if not h:
                    self.assertTrue(
                        (out == pd.Series(["abc", "def"]).str.encode("UTF-8")).all()
                    )


class TestDFFromCSVs(unittest.TestCase):
    """Tests automated loading of CSVs into a pd.DataFrame."""

    def setUp(self):
        self.dir = temp_path()
        names = ["a", "b", "c"]
        self.files = [self.dir / "{}.csv".format(n) for n in names]
        for n, fn in zip(names, self.files):  # make some csvs
            with open(str(fn), "w") as f:
                f.write("C1,C{}\n{},{}\n{},{}".format(n, n, n, n, n))

    def normal_frame_generation(self):
        df = df_from_csvs(self.files)
        expect_cols = ["C1", "Ca", "Cb", "Cc"]
        self.assertIn("C1", df.columns)
        self.assertTrue(len(df.columns) == len(expect_cols))
        self.assertTrue((df.columns == np.array(expect_cols)).all())

    def tearDown(self):
        remove_tempdir(self.dir)


if __name__ == "__main__":
    unittest.main()
