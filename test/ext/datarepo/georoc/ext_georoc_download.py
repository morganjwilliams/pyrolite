import unittest
import os
import time
import pandas as pd
from pyrolite.util.web import internet_connection
from pyrolite.util.general import temp_path, remove_tempdir
from pyrolite.ext.datarepo.georoc.download import (
    get_filelinks,
    update_filelist,
    bulk_download,
)


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestGetGEOROCLinks(unittest.TestCase):
    def test_get_links(self):
        links = get_filelinks(exclude=[])
        self.assertIn("Minerals", links.keys())

    def test_exclude(self):
        for exclude in [["Minerals"], ["Minerals", "Rocks"]]:
            with self.subTest(exclude=exclude):
                links = get_filelinks(exclude=exclude)
                for i in exclude:
                    self.assertNotIn(i, links.keys())


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestUpdateGEOROCFilelist(unittest.TestCase):
    def setUp(self):
        self.temp_dir = temp_path()
        self.filepath = self.temp_dir / "contents.json"
        with open(str(self.filepath), "w+") as fh:
            pass
        self.initial_last_modification_time = os.stat(str(self.filepath)).st_mtime

    def test_update_filelist(self):
        update_filelist(filepath=self.filepath)
        time.sleep(2)  # sleep two seconds to allow updating
        new_modification_time = os.stat(str(self.filepath)).st_mtime
        self.assertTrue(self.filepath.exists())
        self.assertTrue(new_modification_time > self.initial_last_modification_time)

    def tearDown(self):
        remove_tempdir(str(self.filepath.parent))


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestBulkGEOROCCompilation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = temp_path()
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True)
        self.res = ["OBFB"]

    def test_default(self):
        # to csv
        bulk_download(output_folder=self.temp_dir, collections=self.res)

    def tearDown(self):
        remove_tempdir(self.temp_dir)

if __name__ == "__main__":
    unittest.main()
