import unittest
from pyrolite.util.web import internet_connection, download_file
from pyrolite.util.repositories.georoc import parse_GEOROC_response, georoc_munge
from pyrolite.geochem.parse import check_multiple_cation_inclusion


@unittest.skipIf(not internet_connection(), "Needs internet connection.")
class TestGEOROCMunge(unittest.TestCase):
    def setUp(self):
        self.test_url = (
            "http://"
            + "georoc.mpch-mainz.gwdg.de/georoc/Csv_Downloads/"
            + "Complex_Volcanic_Settings_comp/"
            + "FINGER_LAKES_FIELD_NEW_YORK.csv"
        )

        self.df = download_file(
            self.test_url, encoding="latin-1", postprocess=parse_GEOROC_response
        )

    def test_munge(self):
        out = georoc_munge(self.df)

        for c in ["Lat", "Long", "GeolAge"]:
            self.assertIn(c, out.columns)

        self.assertNotIn("Ti", check_multiple_cation_inclusion(out))


if __name__ == "__main__":
    unittest.main()
