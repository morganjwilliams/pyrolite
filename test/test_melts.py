import unittest
import numpy as np
from pyrolite.melts import *


def test_df(cols=['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2'],
            index_length=10):
    return pd.DataFrame({k: v for k,v in zip(cols,
                         np.random.rand(len(cols), index_length))})


def test_ser(index=['SiO2', 'CaO', 'MgO', 'FeO', 'TiO2']):
    return pd.Series({k: v for k,v in zip(index, np.random.rand(len(index)))})


class TestMELTSEnv(unittest.TestCase):

    def setUp(self):
        pass

    def test_env(self):
        env = MELTS_env()



class TestParseMELTSComposition(unittest.TestCase):

    def setUp(self):
        self.cstring = """Fe''0.18Mg0.83Fe'''0.04Al1.43Cr0.52Ti0.01O4"""

    def test_parse(self):
        ret = from_melts_cstr(self.cstring)
        self.assertTrue(isinstance(ret, dict))
        self.assertTrue('Fe2+' in ret.keys())
        self.assertTrue(np.isclose(ret['Fe2+'], 0.18))


class TestMELTSSystem(unittest.TestCase):

    def setUp(self):
        self.ser = test_ser()

    def test_system_build(self):
        ret = MeltsSystem(self.ser)


class TestToMELTSFile(unittest.TestCase):

    def setUp(self):
        self.df = test_df()
        self.ser = test_ser()

    def test_series_to_melts_file(self):
        ret = to_meltsfile(self.ser)

    def test_df_to_melts_file(self):
        ret = to_meltsfile(self.df)


if __name__ == '__main__':
    unittest.main()
