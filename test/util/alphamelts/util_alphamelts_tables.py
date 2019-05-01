import unittest
from pyrolite.util.alphamelts.download import install_melts
from pyrolite.util.meta import pyrolite_datafolder
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.util.alphamelts.tables import get_experiments_summary, MeltsOutput

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

class TestGetExperimentsSummary(unittest.TestCase):
    def setUp(self):
        self.dir = temp_path() / "test_melts_temp"
        self.dir.mkdir()

    def test_default(self):
        pass
        #summary = get_experiments_summary(self.dir)

    def tearDown(self):
        if self.dir.exists():
            remove_tempdir(self.dir)


if __name__ == "__main__":

    unittest.main()
