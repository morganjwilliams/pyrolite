import os, time
import unittest
import pandas as pd
import numpy as np
from pyrolite.util.pd import *
from pathlib import Path


class TestPandasUnitsPatch(unittest.TestCase):

    @unittest.expectedFailure
    def test_classes(self):
        """
        Check patching hasn't occurred to start with.
        """
        for cls in [pd.DataFrame, pd.Series]:
            with self.subTest(cls=cls):
                for prop in [units, # units property
                             set_units, # set_units method
                             ]:
                    with self.subTest(prop=prop):
                        clsp = getattr(cls, prop)
                        instp = getattr(cls(), prop)

    def test_patch(self):
        """
        Check that the units attribute exists after patching.
        """
        patch_pandas_units()
        for cls in [pd.DataFrame, pd.Series]:
            with self.subTest(cls=cls):
                for prop in ['units', # units property
                             'set_units', # set_units method
                             ]:
                    with self.subTest(prop=prop):
                        clsp = getattr(cls, prop)
                        instp = getattr(cls(), prop)

    def test_set_units(self):
        """
        Check that the set_units method works after patching.
        """
        patch_pandas_units()
        test_units = 'Wt%'
        for cls in [pd.DataFrame, pd.Series]:
            with self.subTest(cls=cls):
                instance = cls()
                instance.set_units('Wt%')
                equiv = (instance.units == test_units)
                if not isinstance(equiv, bool):
                    equiv = equiv.all()
                    self.assertTrue(isinstance(instance.units, pd.Series))

                self.assertTrue(equiv)


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
        self.others = [test_df()]*4

    def test_column_order(self):
        result = accumulate([self.df0]+self.others)
        self.assertTrue(all(result.columns == self.df0.columns))

    def test_index_preservation(self):
        result = accumulate([self.df0]+self.others)
        # The range index should just be repeated 5 times, not reset
        self.assertTrue(all([res==exp for (res, exp) in
                             zip(list(result.index.values),
                                 list(np.tile(self.df0.index.values, 5)))]
                            ))


class TestToFrame(unittest.TestCase):

    def setUp(self):
        self.ser = test_ser()
        self.df = test_df()

    def test_df_column_order(self):
        result = to_frame(self.df)
        self.assertTrue(all(result.columns == self.df.columns))

    def test_ser_column_order(self):
        result = to_frame(self.ser)
        self.assertTrue(all(result.columns == self.ser.index))

    def test_df_index_preservation(self):
        result = to_frame(self.df)
        self.assertTrue(all(result.index == self.df.index))

    def test_series_conversion(self):
        result = to_frame(self.ser)
        self.assertTrue(isinstance(result, pd.DataFrame))


class TestToNumeric(unittest.TestCase):

    def setUp(self):
        self.df = test_df().applymap(str)

    def test_numeric(self):
        df = self.df
        result = to_numeric(df)
        self.assertTrue((result.dtypes == 'float64').all())

    def test_exclude(self):
        df = self.df
        exclude = ['TiO2']
        num_columns = [c for c in df.columns if c not in exclude]
        result = to_numeric(df, exclude=exclude)
        print(result, result.dtypes)
        self.assertTrue((result[exclude].dtypes != 'float64').all())
        self.assertTrue((result[num_columns].dtypes == 'float64').all())

    def test_error_methods(self):
        df = self.df
        df.loc[0, 'SiO2'] = 'Low'
        for method in ['ignore', 'raise', 'coerce']:
            with self.subTest(method=method):
                try:
                    result = to_numeric(df, errors=method)
                    self.assertTrue(method in ['ignore', 'coerce'])
                    if method == 'ignore':
                        self.assertTrue(result.loc[0, 'SiO2'] == 'Low')
                    else:
                        self.assertTrue(pd.isnull(result.loc[0, 'SiO2']))
                except ValueError: # should raise with can't parse 'low'
                    self.assertTrue(method=='raise')


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
            try:
                os.remove(self.filename)
            except PermissionError:
                time.sleep(2)
                os.remove(self.filename)

if __name__ == '__main__':
    unittest.main()
