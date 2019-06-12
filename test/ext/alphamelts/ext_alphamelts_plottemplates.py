import unittest
import pandas as pd
from pyrolite.ext.alphamelts.download import install_melts
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.ext.alphamelts.automation import *
from pyrolite.ext.alphamelts.plottemplates import *
from pyrolite.ext.alphamelts.tables import get_experiments_summary
from pyrolite.util.meta import pyrolite_datafolder, stream_log

if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
    stream_log('pyrolite.ext.alphamelts')
    install_melts(local=True)  # install melts for example files etc

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


@unittest.skipIf(not check_perl(), "Perl is not installed.")
class TestTemplates(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / ("test_melts_temp" + self.__class__.__name__)
        self.meltsfile = _melts
        self.envfile = _env  # use default
        title = "MORB"
        # create one experiment folder and run the experiment
        self.folder = make_meltsfolder(
            self.meltsfile, title=title, env=self.envfile, dir=self.dir
        )
        self.process = MeltsProcess(
            meltsfile='{}.melts'.format(title),
            env='environment.txt',
            fromdir=str(self.folder),
        )
        self.process.write([3, 1, 4], wait=True, log=False)
        self.process.terminate()

        self.summary = get_experiments_summary(self.dir)

    def test_plot_phasetable(self):
        ax = plot_phasetable(self.summary)  # phasevol

    def test_plot_comptable(self):
        plot_comptable(self.summary)  # liquidcomp

    def test_plot_phase_composition(self):
        plot_phase_composition(self.summary)  # olivine

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


if __name__ == "__main__":

    unittest.main()
