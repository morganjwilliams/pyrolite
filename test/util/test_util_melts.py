import unittest
from pyrolite.util.general import remove_tempdir
from pyrolite.util.melts import *
from pathlib import Path
from collections import OrderedDict

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

class TestDownload(unittest.TestCase):

    def setUp(self):
        userdir = Path("~").expanduser()
        d, r = userdir.drive / userdir.root
        self.temp_dir = d / 'test_melts_temp'

    def check_download(self):
        """Tries to download MELTS file to a specific directory."""
        download_melts(self.temp_dir)

    def tearDown(self):
        remove_tempdir(self.temp_dir)


class TestInstall(unittest.TestCase):

    def setUp(self):
        userdir = Path("~").expanduser()
        d, r = userdir.drive / userdir.root
        self.temp_dir = d / 'test_melts_temp'
        self.dir = d / 'test_melts_install'

    def check_perl_install(self):
        """Uses subprocess to call the perl installation method."""
        if not check_perl():
            pass
        else:
            for keeptemp in [False, True]:
                with self.subTest(keeptemp=keeptemp):
                    install_melts(self.dir,
                                  native=False,
                                  keep_tempdir=keeptemp)
                    self.assertTrue((self.dir / examples).exists())
                    self.assertTrue((self.dir / 'file_format.command').exists())
                    if keeptemp:
                        self.assertTrue(self.temp_dir.exists() & \
                                        self.temp_dir.is_dir())

    def check_native_install(self):
        """
        Performs the equivalent actions to the perl install script in python.
        """
        for keeptemp in [False, True]:
           with self.subTest(keeptemp=keeptemp):
               install_melts(self.dir,
                             native=True,
                             keep_tempdir=keeptemp)
               self.assertTrue((self.dir / examples).exists())
               self.assertTrue((self.dir / 'file_format.command').exists())
               if keeptemp:
                   self.assertTrue(self.temp_dir.exists() & \
                                   self.temp_dir.is_dir())


    def tearDown(self):
        remove_tempdir(self.dir)
        remove_tempdir(self.temp_dir)


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


if __name__ == '__main__':
    unittest.main()
