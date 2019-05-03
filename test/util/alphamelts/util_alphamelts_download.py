import unittest
from pyrolite.util.web import internet_connection
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.util.alphamelts.download import *


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestDownload(unittest.TestCase):
    """Tests the melts download process."""

    def setUp(self):
        self.temp_dir = temp_path() / "test_melts_temp"

    def check_download(self):
        """Tries to download MELTS files to a specific directory."""
        download_melts(self.temp_dir)
        alphafiles = self.temp_dir.glob("alphamelts*")  # .exe on windows
        self.assertTrue(len(alphafiles))

    def tearDown(self):
        remove_tempdir(self.temp_dir)


class TestInstall(unittest.TestCase):
    """
    Tests the melts install process.

    Todo
    -----
        * Check the executables are copied with exectable permissions where possible
    """

    def setUp(self):
        self.temp_dir = temp_path() / "test_melts_temp"
        self.dir = temp_path() / "test_melts_install"

    @unittest.skipIf(not check_perl(), "Perl is not installed.")
    def test_perl_install(self):
        """Uses subprocess to call the perl installation method."""
        for keeptemp in [False, True]:
            with self.subTest(keeptemp=keeptemp):
                install_melts(
                    self.dir,
                    native=False,
                    temp_dir=self.temp_dir,
                    keep_tempdir=keeptemp,
                )
                self.assertTrue((self.dir / "examples").exists())
                self.assertTrue((self.dir / "file_format.command").exists())
                if keeptemp:
                    self.assertTrue(self.temp_dir.exists() & self.temp_dir.is_dir())

    def test_native_install(self):
        """
        Performs the equivalent actions to the perl install script in python.
        """

        for keeptemp in [False, True]:
            with self.subTest(keeptemp=keeptemp):
                install_melts(
                    self.dir, native=True, temp_dir=self.temp_dir, keep_tempdir=keeptemp
                )
                self.assertTrue((self.dir / "examples").exists())
                self.assertTrue((self.dir / "file_format.command").exists())
                if keeptemp:
                    self.assertTrue(self.temp_dir.exists() & self.temp_dir.is_dir())

    def test_local_install(self):
        """
        Checks that the local install procedure works.
        """
        dir = pyrolite_datafolder() / "alphamelts" / "localinstall"
        for keeptemp in [False, True]:
            with self.subTest(keeptemp=keeptemp):
                install_melts(
                    native=True,
                    temp_dir=self.temp_dir,
                    keep_tempdir=keeptemp,
                    local=True,
                )

                self.assertTrue(dir.exists())
                self.assertTrue((dir / "file_format.command").exists())
                if keeptemp:
                    self.assertTrue(self.temp_dir.exists() & self.temp_dir.is_dir())

    def tearDown(self):
        # may have been removed already, or not created (for local)
        if self.dir.exists():
            remove_tempdir(self.dir)
        if self.temp_dir.exists():
            remove_tempdir(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
