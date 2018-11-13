import unittest
from pathlib import Path
from collections import OrderedDict
from .melts import *
from pyrolite.data.melts.env import MELTS_environment_variables
from pyrolite.util.general import remove_tempdir, internet_connection, \
                                  check_perl, temp_path
from pyrolite.util.pd import test_df, test_ser

def get_default_datadict():
    d = OrderedDict()
    d['title'] = 'TestREST',
    d['initialize'] = {"SiO2": 48.68,
                       "TiO2": 1.01,
                        "Al2O3": 17.64,
                        "Fe2O3": 0.89,
                        "Cr2O3": 0.0425,
                        "FeO": 7.59,
                        "MnO": 0.0,
                        "MgO": 9.10,
                        "NiO": 0.0,
                        "CoO": 0.0,
                        "CaO": 12.45,
                        "Na2O": 2.65,
                        "K2O": 0.03,
                        "P2O5": 0.08,
                        "H2O": 0.20 }
    d['calculationMode']='findLiquidus'
    d['constraints'] = {"setTP": {"initialT": 1200,
                                  "initialP": 1000}}
    return d


class TestParseMELTSComposition(unittest.TestCase):

    def setUp(self):
        self.cstring = """Fe''0.18Mg0.83Fe'''0.04Al1.43Cr0.52Ti0.01O4"""

    def test_parse(self):
        ret = from_melts_cstr(self.cstring)
        self.assertTrue(isinstance(ret, dict))
        self.assertTrue('Fe2+' in ret.keys())
        self.assertTrue(np.isclose(ret['Fe2+'], 0.18))


class TestToMELTSFiles(unittest.TestCase):

    def setUp(self):
        self.df = test_df()
        self.df.loc[:, 'Title'] = ['Title {}'.format(x)
                                    for x in self.df.index.values]
        self.ser = test_ser()
        self.ser.loc['Title'] = 'Test_title'

    def test_series_to_melts_file(self):
        ret = to_meltsfiles(self.ser)

    def test_df_to_melts_files(self):
        ret = to_meltsfiles(self.df)


class TestMELTSEnv(unittest.TestCase):

    def setUp(self):
        self.prefix = 'ALPHAMELTS_'
        self.env_vars = MELTS_environment_variables

    def test_env_build(self):
        """Tests the environment setup with the default config."""
        menv = MELTS_Env(prefix=self.prefix,
                         variable_model=self.env_vars)
        test_var = 'ALPHAMELTS_MINP'
        self.assertTrue(test_var in os.environ)

    def test_valid_setattr(self):
        """Tests that environment variables can be set."""

        menv = MELTS_Env(prefix=self.prefix,
                         variable_model=self.env_vars)
        test_var = 'ALPHAMELTS_MINP'
        for var in [test_var,
                  remove_prefix(test_var, self.prefix)]:
            with self.subTest(var=var):
                for value in [1., 10., 100., 10.]:
                    setattr(menv, var, value)
                    self.assertTrue(test_var in os.environ)
                    self.assertTrue(type(value)(os.environ[test_var])==value)

    def test_reset(self):
        """
        Tests that environment variables can be reset to default/removed
        by setting to None.
        """
        menv = MELTS_Env(prefix=self.prefix,
                         variable_model=self.env_vars)
        test_var = 'ALPHAMELTS_OLD_GARNET'
        for var in [test_var, remove_prefix(test_var, self.prefix)]:
            with self.subTest(var=var):
                setattr(menv, var, True) # set
                setattr(menv, var, None) # reset to default/remove
                _var = remove_prefix(var, self.prefix)
                default = self.env_vars[_var].get('default', None)
                if default is not None:
                    self.assertTrue(type(default)(
                                    os.environ[test_var]
                                    )==default)
                else:
                    self.assertTrue(test_var not in os.environ)


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestWebService(unittest.TestCase):
    """Tests the current MELTS webservice interactivity with default data."""

    def setUp(self):
        self.dict = get_default_datadict()

    def test_melts_compute(self):
        """Tests the MELTS-compute web service."""
        result = melts_compute(self.dict)

    def test_melts_oxides(self):
        """Tests the MELTS-oxides web service."""
        result = melts_oxides(self.dict)

    def test_melts_phases(self):
        """Tests the MELTS-phases web service."""
        result = melts_phases(self.dict)


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestDownload(unittest.TestCase):
    """Tests the melts download process."""

    def setUp(self):
        self.temp_dir = temp_path() / 'test_melts_temp'

    def check_download(self):
        """Tries to download MELTS files to a specific directory."""
        download_melts(self.temp_dir)
        alphafiles = self.temp_dir.glob("alphamelts*") # .exe on windows
        self.assertTrue(len(alphafiles))

    def tearDown(self):
        remove_tempdir(self.temp_dir)


class TestInstall(unittest.TestCase):
    """Tests the melts install process."""

    def setUp(self):
        self.temp_dir = temp_path() / 'test_melts_temp'
        self.dir = temp_path() / 'test_melts_install'


    @unittest.skipIf(not check_perl(), "Perl is not installed.")
    def test_perl_install(self):
        """Uses subprocess to call the perl installation method."""
        userdir = Path("~").expanduser()
        d, r = userdir.drive,  userdir.root
        self.temp_dir = Path(d) / r / 'test_melts_temp'
        self.dir = Path(d) / r / 'test_melts_install'
        for keeptemp in [False, True]:
            with self.subTest(keeptemp=keeptemp):
                install_melts(self.dir,
                              native=False,
                              temp_dir=self.temp_dir,
                              keep_tempdir=keeptemp)
                self.assertTrue((self.dir / 'examples').exists())
                self.assertTrue((self.dir / 'file_format.command').exists())
                if keeptemp:
                    self.assertTrue(self.temp_dir.exists() & \
                                    self.temp_dir.is_dir())

    def test_native_install(self):
        """
        Performs the equivalent actions to the perl install script in python.
        """

        for keeptemp in [False, True]:
           with self.subTest(keeptemp=keeptemp):
               install_melts(self.dir,
                             native=True,
                             temp_dir=self.temp_dir,
                             keep_tempdir=keeptemp)
               self.assertTrue((self.dir / 'examples').exists())
               self.assertTrue((self.dir / 'file_format.command').exists())
               if keeptemp:
                   self.assertTrue(self.temp_dir.exists() & \
                                   self.temp_dir.is_dir())

    def tearDown(self):
        remove_tempdir(self.dir)
        remove_tempdir(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
