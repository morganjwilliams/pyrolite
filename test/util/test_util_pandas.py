import os
import unittest
import pandas as pd
import numpy as np
from pyrolite.util.pandas import *
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
        pass

    def test_column_order(self):
        pass

    def test_index_preservation(self):
        pass


class TestToFrame(unittest.TestCase):

    def setUp(self):
        self.ser = pd.Series()
        self.df = pd.Series()

    def test_column_order(self):
        pass

    def test_index_preservation(self):
        pass

    def test_series_conversion(self):
        result = to_frame(self.ser)
        self.assertTrue(isinstance(result, pd.DataFrame))




class TestToNumeric(unittest.TestCase):

    def setUp(self):
        pass

    def test_exclude(self):
        pass

    def test_error_methods(self):
        pass


class TestConcatColumns(unittest.TestCase):

    def setUp(self):
        pass

    def test_exclude(self):
        pass

    def test_error_methods(self):
        pass


class TestUniquesFromConcat(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame()

    def test_unique(self):
        pass
        uniques_from_concat


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
        self.filename ='tst_save_pickle.pkl'

    def test_pickling(self):
        sparse_pickle_df(self.df, self.filename)
        file = Path(self.filename)
        for cond in [file.exists(), file.is_file()]:
            with self.subTest(cond=cond):
                self.assertTrue(cond)

    def tearDown(self):
        # Delete temporary files
        os.remove(self.filename)


class TestLoadSparsePickleDF(unittest.TestCase):
    """Test the pickle loader."""

    def setUp(self):
        # create test file
        self.df = pd.DataFrame()
        self.filename = 'tst_load_pickle.pkl'

    def test_load(self):
        """Test loading a dataframe."""

        sparse_pickle_df(self.df, self.filename)
        try:
            for keep_sparse in [True, False]:
                with self.subTest(keep_sparse=keep_sparse):
                    df = load_sparse_pickle_df(self.filename,
                                               keep_sparse=keep_sparse)
                    if keep_sparse:
                        self.assertTrue(isinstance(df, pd.SparseDataFrame))
        finally:
            os.remove(self.filename)

if __name__ == '__main__':
    unittest.main()
