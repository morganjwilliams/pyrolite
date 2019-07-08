import io
import unittest
import pandas as pd
from pyrolite.util.pd import to_numeric
from pyrolite.util.meta import pyrolite_datafolder, stream_log
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.geochem.norm import ReferenceCompositions
from pyrolite.ext.alphamelts.download import install_melts
from pyrolite.ext.alphamelts.automation import *
import logging

logger = logging.Logger(__name__)
stream_log(logger)
stream_log('pyrolite.ext.alphamelts.automation', level='DEBUG')

_env = MELTS_Env()
_env.VERSION = "MELTS"
_env.MODE = "isobaric"
_env.MINP = 2000
_env.MAXP = 10000
_env.MINT = 500
_env.MAXT = 1500
_env.DELTAT = -10
_env.DELTAP = 0

Gale_MORB = ReferenceCompositions()["MORB_Gale2013"]
MORB = Gale_MORB.original_data.loc[
    [
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
    ],
    "value",
].apply(pd.to_numeric)
MORB = pd.DataFrame([MORB, MORB]).reset_index().drop(columns="index")
MORB["Title"] = [
    "{}-{}".format(Gale_MORB.ModelName, ix)
    for ix in MORB.index.values.astype(str)
]

if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
    stream_log("pyrolite.ext.alphamelts")
    install_melts(local=True)  # install melts for example files etc

with open(
    (
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
        folder = make_meltsfolder(
            self.meltsfile, "MORB", env=self.env, dir=self.dir
        )

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
        title = "MORB"
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
            meltsfile=self.meltsfile, title="Experiment", env=self.env, dir=self.dir
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
