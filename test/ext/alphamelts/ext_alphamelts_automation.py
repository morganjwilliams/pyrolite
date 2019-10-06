import io
import unittest
import pandas as pd
from pyrolite.util.pd import to_numeric
from pyrolite.util.meta import pyrolite_datafolder, stream_log
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.geochem.norm import get_reference_composition
from pyrolite.ext.alphamelts.download import install_melts
from pyrolite.ext.alphamelts.automation import *
import logging

logger = logging.Logger(__name__)

if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
    stream_log("pyrolite.ext.alphamelts")
    install_melts(local=True)  # install melts for example files etc

_env = MELTS_Env()
_env.VERSION = "MELTS"
_env.MODE = "isobaric"
_env.MINP = 2000
_env.MAXP = 10000
_env.MINT = 500
_env.MAXT = 1500
_env.DELTAT = -10
_env.DELTAP = 0

with open(
    str(
        pyrolite_datafolder(subfolder="alphamelts")
        / "localinstall"
        / "examples"
        / "Morb.melts"
    )
) as f:
    _melts = f.read()


class TestMakeMeltsFolder(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.env = _env  # use default

    def test_default(self):
        folder = make_meltsfolder(self.meltsfile, "MORB", env=self.env, dir=self.dir)

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsProcess(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.env = _env  # use default

    def test_default(self):
        title = "TestMeltsProcess"
        folder = make_meltsfolder(
            self.meltsfile, title=title, env=self.env, dir=self.dir
        )
        process = MeltsProcess(
            meltsfile="{}.melts".format(title),
            env="environment.txt",
            fromdir=str(folder),
        )
        txtfiles = list(self.dir.glob("**/*.txt"))
        meltsfiles = list(self.dir.glob("**/*.melts"))
        process.write([3, 1, 4], wait=True, log=False)
        process.terminate()

    def tearDown(self):
        if self.dir.exists():
            try:
                remove_tempdir(self.dir)
            except FileNotFoundError:
                pass


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsExperiment(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.env = _env  # use default

    def test_default(self):
        exp = MeltsExperiment(
            meltsfile=self.meltsfile,
            title="TestMeltsExperiment",
            env=self.env,
            dir=self.dir,
        )
        # check the folder has been created correctly
        txtfiles = list(self.dir.glob("**/*.txt"))
        meltsfiles = list(self.dir.glob("**/*.melts"))
        exp.run()
        exp.cleanup()

    def tearDown(self):
        if self.dir.exists():
            try:
                remove_tempdir(self.dir)
            except FileNotFoundError:
                pass


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestMeltsBatch(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)

        Gale_MORB = get_reference_composition("MORB_Gale2013")
        majors = [
            "SiO2",
            "Al2O3",
            "FeO",
            "MnO",
            "MgO",
            "CaO",
            "Na2O",
            "TiO2",
            "K2O",
            "P2O5",
        ]
        MORB = Gale_MORB.comp.loc[:, majors].apply(pd.to_numeric)
        MORB = MORB.append(MORB).reset_index().drop(columns="index")
        MORB["Title"] = [
            "{}_{}".format(Gale_MORB.name, ix).replace("_", "")
            for ix in MORB.index.values.astype(str)
        ]
        MORB["Initial Temperature"] = 1300
        MORB["Final Temperature"] = 800
        MORB["Initial Pressure"] = 5000
        MORB["Final Pressure"] = 5000
        MORB["Log fO2 Path"] = "FMQ"
        MORB["Increment Temperature"] = -5
        MORB["Increment Pressure"] = 0
        self.df = MORB
        self.env = _env

    def test_default(self):
        batch = MeltsBatch(
            self.df,
            default_config={
                "Initial Pressure": 7000,
                "Initial Temperature": 1400,
                "Final Temperature": 800,
                "modes": ["isobaric"],
            },
            grid={"Initial Pressure": [5000]},
            env=self.env,
            fromdir=self.dir,
            logger=logger,
        )
        batch.run()

    def tearDown(self):
        if self.dir.exists():
            try:
                remove_tempdir(self.dir)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
