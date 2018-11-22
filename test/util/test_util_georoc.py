import os
import unittest
import pandas as pd
from pyrolite.util.georoc import *
from pyrolite.geochem import check_multiple_cation_inclusion
from pyrolite.util.general import internet_connection, temp_path, remove_tempdir


class TestParseValues(unittest.TestCase):
    def test_single_entry(self):
        for val, expect in [("A [1]", "A")]:
            with self.subTest(val=val, expect=expect):
                self.assertEqual(parse_values(val), expect)

    def test_series(self):
        for val, expect in [(pd.Series(["A [1]"]), pd.Series(["A"]))]:
            with self.subTest(val=val, expect=expect):
                self.assertTrue((parse_values(val) == expect).all())

    def test_sub(self):
        for val, sub, expect in [
            ("A [1]", lambda x: x, "A"),
            ("A, [1]", subsitute_commas, "A"),
        ]:
            with self.subTest(val=val, sub=sub, expect=expect):
                self.assertEqual(parse_values(val, sub=sub), expect)


class TestParseCitations(unittest.TestCase):
    def setUp(self):
        pass

    def test_single_entry(self):
        for val, expect in [("[1] A", {"value": "A", "key": "1"})]:
            with self.subTest(val=val, expect=expect):
                self.assertEqual(parse_citations(val), expect)

    def test_series(self):
        for val, expect in [
            (pd.Series(["[1] A"]), pd.Series([{"value": "A", "key": "1"}]))
        ]:
            with self.subTest(val=val, expect=expect):
                self.assertTrue((parse_citations(val) == expect).all())


class TestParseDOI(unittest.TestCase):
    def test_single_entry(self):
        for val, expect in [("[1] A doi:10.x01/g", "dx.doi.org/10.x01/g")]:
            with self.subTest(val=val, expect=expect):
                self.assertEqual(parse_DOI(val), expect)

    def test_series(self):
        for val, expect in [
            (pd.Series(["[1] A doi:10.x01/g"]), pd.Series(["dx.doi.org/10.x01/g"]))
        ]:
            with self.subTest(val=val, expect=expect):
                self.assertTrue((parse_DOI(val) == expect).all())

    def test_link(self):
        for val, link, expect in [
            ("[1] A doi:10.x01/g", True, "dx.doi.org/10.x01/g"),
            ("[1] A doi:10.x01/g", False, "10.x01/g"),
        ]:
            with self.subTest(val=val, link=link, expect=expect):
                self.assertEqual(parse_DOI(val, link=link), expect)


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestGetGEOROCLinks(unittest.TestCase):
    def test_get_links(self):
        links = get_georoc_links(exclude=[])
        self.assertIn("Minerals", links.keys())

    def test_exclude(self):
        for exclude in [["Minerals"], ["Minerals", "Rocks"]]:
            with self.subTest(exclude=exclude):
                links = get_georoc_links(exclude=exclude)
                for i in exclude:
                    self.assertNotIn(i, links.keys())


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestUpdateGEOROCFilelist(unittest.TestCase):
    def setUp(self):
        self.temp_dir = temp_path() / "test_pyrolite.util.georoc"
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True)
        self.filepath = self.temp_dir / "contents.json"
        with open(str(self.filepath), "w+") as fh:
            pass
        self.initial_last_modification_time = os.stat(str(self.filepath)).st_mtime

    def test_update_filelist(self):
        update_georoc_filelist(filepath=self.filepath)
        new_modification_time = os.stat(str(self.filepath)).st_mtime
        self.assertTrue(self.filepath.exists())
        self.assertTrue(new_modification_time > self.initial_last_modification_time)

    def tearDown(self):
        remove_tempdir(str(self.filepath.parent))


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestDownloadGEOROCCompilation(unittest.TestCase):
    def setUp(self):
        self.test_url = (
            "http://"
            + "georoc.mpch-mainz.gwdg.de/georoc/Csv_Downloads/"
            + "Complex_Volcanic_Settings_comp/"
            + "FINGER_LAKES_FIELD_NEW_YORK.csv"
        )

    def test_dataframe_return(self):
        df = download_GEOROC_compilation(self.test_url)


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestBulkGEOROCCompilation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = temp_path() / "test_pyrolite.util.georoc"
        if not self.temp_dir.exists():
            self.temp_dir.mkdir(parents=True)
        self.res = ["OBFB"]

    def test_dataframe_return(self):
        bulk_GEOROC_download(output_folder=self.temp_dir, reservoirs=self.res)

    def tearDown(self):
        remove_tempdir(self.temp_dir)


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestGEOROCMunge(unittest.TestCase):
    def setUp(self):
        self.test_url = (
            "http://"
            + "georoc.mpch-mainz.gwdg.de/georoc/Csv_Downloads/"
            + "Complex_Volcanic_Settings_comp/"
            + "FINGER_LAKES_FIELD_NEW_YORK.csv"
        )

        self.df = download_GEOROC_compilation(self.test_url)

    def test_munge(self):
        out = georoc_munge(self.df)

        for c in ["Lat", "Long", "GeolAge"]:
            self.assertIn(c, out.columns)

        self.assertNotIn("Ti", check_multiple_cation_inclusion(out))


if __name__ == "__main__":
    unittest.main()
