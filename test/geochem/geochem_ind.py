import unittest
from pyrolite.geochem.ind import *


class TestGetCations(unittest.TestCase):
    """Tests the cation calculator."""

    def test_none(self):
        """Check the function works for no cations."""
        for cationstring in ["O", "O2", ""]:
            with self.subTest(cationstring=cationstring):
                self.assertTrue(len(get_cations(cationstring)) == 0)

    def test_single(self):
        """Check the function works for a single cation."""
        for cationstring in ["SiO2", "MgO", "Si"]:
            with self.subTest(cationstring=cationstring):
                self.assertTrue(len(get_cations(cationstring)) == 1)

    def test_multiple(self):
        """Check the function works for multiple cations."""
        for cationstring in ["MgSiO2", "MgSO4", "CaCO3", "Na2Mg3Al2Si8O22(OH)2"]:
            with self.subTest(cationstring=cationstring):
                self.assertTrue(len(get_cations(cationstring)) > 1)

    def test_exclude(self):
        """Checks that the exclude function works."""
        for ox, excl in [
            ("MgO", ["O"]),
            ("MgO", []),
            ("MgSO4", ["O", "S"]),
            ("MgSO4", ["S"]),
            ("Mg(OH)2", ["O", "H"]),
            ("Mg(OH)2", ["H"]),
        ]:
            with self.subTest(ox=ox, excl=excl):
                self.assertTrue(len(get_cations(ox, exclude=excl)) == 1)


class TestCommonElements(unittest.TestCase):
    """Tests the common element generator."""

    def test_cutoff(self):
        """Check the function works normal cutoff Z numbers."""
        for cutoff in [1, 15, 34, 63, 93]:
            with self.subTest(cutoff=cutoff):
                self.assertTrue(
                    common_elements(output="formula", cutoff=cutoff)[-1].number
                    == cutoff
                )

    def test_high_cutoff(self):
        """Check the function works silly high cutoff Z numbers."""
        for cutoff in [119, 1000, 10000]:
            with self.subTest(cutoff=cutoff):
                self.assertTrue(
                    len(common_elements(output="formula", cutoff=cutoff)) < 130
                )
                self.assertTrue(
                    common_elements(output="formula", cutoff=cutoff)[-1].number < cutoff
                )

    def test_formula_output(self):
        """Check the function produces formula output."""
        for el in common_elements(cutoff=10, output="formula"):
            with self.subTest(el=el):
                self.assertIs(type(el), type(pt.elements[0]))

    def test_string_output(self):
        """Check the function produces string output."""
        for el in common_elements(cutoff=10, output="string"):
            with self.subTest(el=el):
                self.assertIs(type(el), str)


class TestREE(unittest.TestCase):
    """Tests the Rare Earth Element generator."""

    def setUp(self):
        self.min_z = 57
        self.max_z = 71

    def test_complete(self):
        """Check all REE are present."""
        reels = REE(output="formula", dropPm=False)
        ns = [el.number for el in reels]
        for n in range(self.min_z, self.max_z + 1):
            with self.subTest(n=n):
                self.assertTrue(n in ns)

    def test_precise(self):
        """Check that only the REE are returned."""
        reels = REE(output="formula")
        ns = [el.number for el in reels]
        self.assertTrue(min(ns) == self.min_z)
        self.assertTrue(max(ns) == self.max_z)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for el in REE(output="formula"):
            with self.subTest(el=el):
                self.assertIs(type(el), type(pt.elements[0]))

    def test_string_output(self):
        """Check the function produces string output."""
        for el in REE(output="string"):
            with self.subTest(el=el):
                self.assertIs(type(el), str)


class TestREY(unittest.TestCase):
    """Tests the Rare Earth Element + Yttrium generator."""

    def setUp(self):
        self.min_z = 57
        self.max_z = 71

    def test_complete(self):
        """Check all REE and Yttrium are present."""
        reels = REY(output="formula", dropPm=False)
        ns = [el.number for el in reels]
        print(ns)
        for n in [39] + [*range(self.min_z, self.max_z + 1)]:
            with self.subTest(n=n):
                self.assertTrue(n in ns)

    def test_precise(self):
        """Check that only the REY are returned."""
        reels = REY(output="formula")
        ns = [el.number for el in reels]
        self.assertTrue(min(ns) == 39)
        self.assertTrue(max(ns) == self.max_z)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for el in REY(output="formula"):
            with self.subTest(el=el):
                self.assertIs(type(el), type(pt.elements[0]))

    def test_string_output(self):
        """Check the function produces string output."""
        for el in REY(output="string"):
            with self.subTest(el=el):
                self.assertIs(type(el), str)


class TestSimpleOxides(unittest.TestCase):
    """Tests the simple oxide generator."""

    @unittest.expectedFailure
    def test_none(self):
        """Check the function returns no oxides for no elements in."""
        simple_oxides("", output="formula")

    def test_one(self):
        """Check the function returns oxides for one element in."""
        self.assertTrue(len(simple_oxides("Si", output="formula")) >= 1)

    def test_formula_output(self):
        """Check the function produces formula output."""
        for ox in simple_oxides("Si", output="formula"):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), type(pt.formula("SiO2")))

    def test_string_output(self):
        """Check the function produces string output."""
        for ox in simple_oxides("Si", output="string"):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), str)


class TestCommonOxides(unittest.TestCase):
    """Tests the common oxide generator."""

    def test_none(self):
        """Check the function returns no oxides for no elements in."""
        # When not passed elements, this function uses elements up to Uranium
        # to generate oxides instead.
        out = common_oxides(elements=[], output="formula")
        self.assertTrue(len(out) != 1)

    def test_one(self):
        """Check the function returns oxides for one element in."""
        els = ["Si"]
        out = common_oxides(elements=els, output="formula")
        self.assertTrue(len(out) >= 1)
        for ox in out:
            with self.subTest(ox=ox):
                # All oxides are from elements contained in the list
                self.assertIn(get_cations(ox)[0].__str__(), els)

    def test_multiple(self):
        """Check the function returns oxides for muliple elements in."""
        els = ["Si", "Mg", "Ca"]
        out = common_oxides(elements=els, output="formula")
        self.assertTrue(len(out) >= len(els))
        for ox in out:
            with self.subTest(ox=ox):
                # All oxides are from elements contained in the list
                self.assertIn(get_cations(ox)[0].__str__(), els)

    @unittest.expectedFailure
    def test_invalid_elements(self):
        """Check the function fails for invalid input."""
        not_els = [["SiO2"], ["notanelement"], ["Ci"]]
        for els in not_els:
            with self.subTest(els=els):
                common_oxides(elements=els, output="formula")

    def test_formula_output(self):
        """Check the function produces formula output."""
        for ox in common_oxides(output="formula"):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), type(pt.formula("SiO2")))

    def test_string_output(self):
        """Check the function produces string output."""
        for ox in common_oxides(output="string"):
            with self.subTest(ox=ox):
                self.assertIs(type(ox), str)

    def test_addition(self):
        """Checks the addition functionality."""
        pass

    # As stands, unless addition == [], for string output extras are returned
    def test_precise(self):
        """Check that only relevant oxides are returned."""
        for els in [["Li"], ["Ca", "Ti"], ["Li", "Mg", "K"]]:
            with self.subTest(els=els):
                for ox in common_oxides(elements=els, output="formula"):
                    # All oxides are from elements contained in the list
                    self.assertIn(get_cations(ox)[0].__str__(), els)


class TestGetIonicRadii(unittest.TestCase):
    """Checks the Shannon radii getter."""

    def setUp(self):
        self.ree = REE()

    def test_ree_radii(self):
        radii = get_ionic_radii(self.ree[0], charge=3, coordination=8)
        self.assertTrue(isinstance(radii, float))

    def test_ree_radii_list(self):
        radii = get_ionic_radii(self.ree, charge=3, coordination=8)
        self.assertTrue(isinstance(radii, list))

    def test_ree_radii_list_whittaker_muntus(self):
        radii = get_ionic_radii(self.ree, charge=3, coordination=8, source="whittaker")
        self.assertTrue(isinstance(radii, list))


class TestByIncompatibility(unittest.TestCase):
    def setUp(self):
        self.els = ["Ni", "Mg", "Cs", "Sr"]

    def test_default(self):
        reordered_REE = by_incompatibility(self.els)
        self.assertEqual(reordered_REE[0], "Cs")

    def test_reverse(self):
        reordered_REE = by_incompatibility(self.els, reverse=True)
        self.assertEqual(reordered_REE[0], "Ni")


class TestByNumber(unittest.TestCase):
    def setUp(self):
        self.els = ["Ni", "Mg", "Cs", "Sr"]

    def test_default(self):
        reordered_REE = by_number(self.els)
        self.assertEqual(reordered_REE[0], "Mg")
        self.assertEqual(reordered_REE[-1], "Cs")

    def test_reverse(self):
        reordered_REE = by_number(self.els, reverse=True)
        self.assertEqual(reordered_REE[0], "Cs")
        self.assertEqual(reordered_REE[-1], "Mg")

# todo: get_cations


if __name__ == "__main__":
    unittest.main()
