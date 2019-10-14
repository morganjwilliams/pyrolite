import unittest
import pandas as pd
from pyrolite.util.general import temp_path, remove_tempdir
from pyrolite.mineral.mindb import (
    list_groups,
    list_minerals,
    list_formulae,
    get_mineral,
    parse_composition,
    get_mineral_group,
    update_database,
    __dbpath__,
)

update_database()
class TestDBLists(unittest.TestCase):
    def setUp(self):
        pass

    def test_list_minerals(self):
        out = list_minerals()
        self.assertIsInstance(out, list)
        self.assertIn("forsterite", out)

    def test_list_groups(self):
        out = list_groups()
        self.assertIsInstance(out, list)
        self.assertIn("olivine", out)

    def test_list_formulae(self):
        out = list_formulae()
        self.assertIsInstance(out, list)
        self.assertIn("Mg2SiO4", out)


class TestGetMineralGroup(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_olivine(self):
        out = get_mineral_group("olivine")
        self.assertIsInstance(out, pd.DataFrame)

    @unittest.expectedFailure
    def test_get_olivine(self):
        out = get_mineral_group("tourmaline")
        self.assertIsInstance(out, pd.DataFrame)


class TestGetMineral(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_mineral(self):
        for get in ["forsterite", "enstatite"]:
            with self.subTest(get=get):
                out = get_mineral(get)
                self.assertIsInstance(out, pd.Series)

    @unittest.expectedFailure
    def test_non_mineral(self):
        for get in ["andychristyite", "Not quite a formula"]:
            with self.subTest(get=get):
                out = get_mineral(get)
                self.assertIsInstance(out, pd.Series)


class TestParseComposition(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_mineral(self):
        for get in ["forsterite", "enstatite", "Mg2SiO4"]:
            with self.subTest(get=get):
                out = parse_composition(get)
                self.assertIsInstance(out, pd.Series)

    @unittest.expectedFailure
    def test_non_mineral(self):
        for get in ["andychristyite", "Not quite a formula"]:
            with self.subTest(get=get):
                out = parse_composition(get)
                self.assertIsInstance(out, pd.Series)


class TestUpdateDB(unittest.TestCase):
    def setUp(self):
        self.path = __dbpath__
        self.dir = temp_path()
        self.alternatepath = self.dir / __dbpath__.name

    def test_default(self):
        update_database()

    def test_alternate_path(self):
        update_database(path=self.alternatepath)

    def tearDown(self):
        remove_tempdir(self.dir)
