import io
import unittest
import pandas as pd
from pyrolite.util.pd import to_numeric
from pyrolite.util.synthetic import test_df, test_ser
from pyrolite.util.alphamelts.download import install_melts
from pyrolite.util.meta import pyrolite_datafolder
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.util.alphamelts.automation import *

_env = (
    pyrolite_datafolder(subfolder="alphamelts")
    / "localinstall"
    / "examples"
    / "alphamelts_default_env.txt"
)

_melts = (
    pyrolite_datafolder(subfolder="alphamelts")
    / "localinstall"
    / "examples"
    / "morb.melts"
)


class TestMakeMeltsFolder(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / "test_melts_temp"
        self.meltsfile = _melts
        self.envfile = _env  # use default

    def test_default(self):
        path = make_meltsfolder(self.meltsfile, "MORB", env=self.envfile, dir=self.dir)

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsProcess(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / "test_melts_temp"

    def test_default(self):
        MeltsProcess

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsExperiment(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / "test_melts_temp"

    def test_default(self):
        MeltsExperiment

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsBatch(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / "test_melts_temp"

    def test_default(self):
        MeltsBatch

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


if __name__ == "__main__":
    if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
        install_melts(local=True)  # install melts for example files etc

    unittest.main()
