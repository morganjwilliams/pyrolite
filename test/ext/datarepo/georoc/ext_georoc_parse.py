import unittest
import pandas as pd
from pyrolite.util.web import internet_connection
from pyrolite.ext.datarepo.georoc.parse import (
    parse_DOI,
    parse_values,
    parse_citations,
    subsitute_commas,
)


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


if __name__ == "__main__":
    unittest.main()
