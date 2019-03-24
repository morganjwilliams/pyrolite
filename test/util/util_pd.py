import os, time
import unittest
import pandas as pd
import numpy as np
from pyrolite.util.synthetic import test_df, test_ser
from pyrolite.util.pd import *
from pathlib import Path


class TestColumnOrderedAppend(unittest.TestCase):
    def setUp(self):
        pass

    def test_column_order(self):
        pass

    def test_index_preservation(self):
        pass


class TestAccumulate(unittest.TestCase):
    def setUp(self):
        self.df0 = test_df()
        self.others = [test_df()] * 4

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
        self.ser = test_ser()
        self.df = test_df()

    def test_df_column_order(self):
        result = to_frame(self.df)
        self.assertTrue((result.columns == self.df.columns).all())

    def test_ser_column_order(self):
        result = to_frame(self.ser)
        self.assertTrue((result.columns == self.ser.index).all())

    def test_df_index_preservation(self):
        result = to_frame(self.df)
        self.assertTrue((result.index == self.df.index).all())

    def test_series_conversion(self):
        result = to_frame(self.ser)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_non_convertable(self):
        for noconv in [None, [0, 1, [1]]]:
            with self.subTest(noconv=noconv):
                with self.assertRaises(NotImplementedError) as cm:
                    result = to_frame(noconv)


class TestToSer(unittest.TestCase):
    def setUp(self):
        self.ser = test_ser()
        self.df = test_df()

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
        self.df = test_df().applymap(str)

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
        self.df = test_df()

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
        self.df = None

    def test_df_generation(self):
        pass
        df_from_csvs


class TestPickleFromCSVS(unittest.TestCase):
    """Test the CSV pickler."""

    def setUp(self):
        self.dfs = [pd.DataFrame(), pd.DataFrame]


class TestSparsePickleDF(unittest.TestCase):
    """Test the pickler."""

    def setUp(self):
        self.df = pd.DataFrame()
        self.filename = "tst_save_pickle.pkl"

    def test_pickling(self):
        sparse_pickle_df(self.df, self.filename)
        file = Path(self.filename)
        for cond in [file.exists(), file.is_file()]:
            with self.subTest(cond=cond):
                self.assertTrue(cond)

    def tearDown(self):
        try:
            os.remove(self.filename)
        except PermissionError:
            time.sleep(2)
            os.remove(self.filename)


class TestLoadSparsePickleDF(unittest.TestCase):
    """Test the pickle loader."""

    def setUp(self):
        # create test file
        self.df = pd.DataFrame()
        self.filename = "tst_load_pickle.pkl"

    def test_load(self):
        """Test loading a dataframe."""

        sparse_pickle_df(self.df, self.filename)
        try:
            for keep_sparse in [True, False]:
                with self.subTest(keep_sparse=keep_sparse):
                    df = load_sparse_pickle_df(self.filename, keep_sparse=keep_sparse)
                    if keep_sparse:
                        self.assertTrue(isinstance(df, pd.SparseDataFrame))
        finally:
            try:
                os.remove(self.filename)
            except PermissionError:
                time.sleep(2)
                os.remove(self.filename)


if __name__ == "__main__":
    unittest.main()
