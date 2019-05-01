import io
import unittest
import pandas as pd
from pyrolite.util.pd import to_numeric
from pyrolite.util.synthetic import test_df, test_ser
from pyrolite.util.alphamelts.download import install_melts
from pyrolite.util.meta import pyrolite_datafolder
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.util.alphamelts.automation import *
from pyrolite.util.meta import stream_log

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
    / "Morb.melts"
)

if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
    install_melts(local=True)  # install melts for example files etc


class TestMakeMeltsFolder(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.envfile = _env  # use default

    def test_default(self):
        folder = make_meltsfolder(
            self.meltsfile, "MORB", env=self.envfile, dir=self.dir
        )

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsProcess(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.envfile = _env  # use default

    def test_default(self):
        folder = make_meltsfolder(
            self.meltsfile, "MORB", env=self.envfile, dir=self.dir
        )
        process = MeltsProcess(
            meltsfile=self.meltsfile, env=self.envfile, fromdir=str(folder)
        )
        process.write(3, 1, 4, wait=True, log=False)
        process.terminate()

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsExperiment(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.envfile = _env  # use default

    def test_default(self):
        exp = MeltsExperiment(meltsfile=self.meltsfile, env=self.envfile, dir=self.dir)
        exp.run()
        exp.cleanup()

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsBatch(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)

    def test_default(self):
        batch = MeltsBatch()

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


if __name__ == "__main__":
    unittest.main()
