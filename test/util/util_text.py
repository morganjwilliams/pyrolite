import unittest

import numpy as np

from pyrolite.util.text import (
    int_to_alpha,
    normalise_whitespace,
    parse_entry,
    quoted_string,
    remove_prefix,
    slugify,
    split_records,
    string_variations,
    titlecase,
    to_width,
)


class TestRemovePrefix(unittest.TestCase):
    def test_prefix_present(self):
        self.assertEqual(remove_prefix("prefixword", "prefix"), "word")

    def test_prefix_notpresent(self):
        self.assertEqual(remove_prefix("prefixword", "refix"), "prefixword")

    def test_double_prefix(self):
        """Should just remove one prefix."""
        self.assertEqual(remove_prefix("prefixprefixword", "prefix"), "prefixword")


class TestNormaliseWhitespace(unittest.TestCase):
    def test_whitepace_removal(self):
        self.assertEqual(normalise_whitespace("a\n\n\nb"), "a b")

    def test_whitespace_preservation(self):
        self.assertEqual(normalise_whitespace("a b"), "a b")


class TestToWidth(unittest.TestCase):
    def test_width_spec(self):
        """
        Check that the output width is as specifed.
        For strings including whitespace, it should
        preserve word structure.
        """

        s = "*- " * 100
        for width in [1, 10, 79]:
            with self.subTest(width=width):
                out = to_width(s, width=width)
                w = len(out.splitlines()[0])
                self.assertTrue(w <= width)
                self.assertTrue(w >= width - len(s))


class TestQuotedString(unittest.TestCase):
    """Tests for quoted string utility."""

    def test_quoted_string(self):
        """Check that quoted strings operate correctly."""

        for string in [
            "singlephrase",
            "double phrase",
            "tri ple phrase",
            "singlephrase",
            "double phrase",
            "tri ple phrase",
            """singlephrase""",
            """double phrase""",
            """tri ple phrase""",
        ]:
            with self.subTest(string=string):
                quotes = ["'", '"']
                stringcontent = "".join(
                    i for i in quoted_string(string) if i not in quotes
                )
                # Check same content
                self.assertEqual(stringcontent, "".join(i for i in string))

                contains = [[s in q for s in quoted_string(string)] for q in quotes]
                # Check there's at least two of the same quotes.
                self.assertTrue((np.array(contains).sum() >= 2).any())


class TestTitlecase(unittest.TestCase):
    """Tests for titlecase string utility."""

    def test_single_word(self):
        """Check single word cases operate correctly."""
        for string in ["lowercase", "UPPERCASE", "CamelCase", "Titlecase"]:
            with self.subTest(string=string):
                # String content doesnt change
                self.assertEqual(titlecase(string).lower(), string.lower())
                # Single word capitalisation
                self.assertEqual(
                    titlecase(string), string[0].upper() + string[1:].lower()
                )

    def test_multi_word(self):
        """Check multiword case operates correctly."""
        for string in ["lower case", "UPPER CASE", "Camel Case", "Title case"]:
            with self.subTest(string=string):
                # String content doesnt change
                self.assertEqual(
                    titlecase(string).lower(), string.lower().replace(" ", "")
                )

                expected = "".join(
                    [st[0].upper() + st[1:].lower() for st in string.split(" ")]
                )
                self.assertEqual(titlecase(string), expected)

    def test_valid_split_delimiters(self):
        """Check split delimiters operate correctly."""
        for string in [
            "lower case",
            "lower-case",
            "lower_case",
            "lower\tcase",
            "lower\ncase",
        ]:
            with self.subTest(string=string):
                self.assertEqual(titlecase(string).lower(), "lowercase")

    @unittest.expectedFailure
    def test_invalid_split_delimiters(self):
        """Check split delimiters operate correctly."""
        for string in ["lower/case", r"lower\case", "lower&case"]:
            with self.subTest(string=string):
                self.assertEqual(titlecase(string).lower(), "lowercase")

    def test_initial_space(self):
        """Check that strings are stripped effectively."""
        # this is failing!
        # self.assertEqual(titlecase(" a_b"), "AB")

    def test_join_characters(self):
        """Check join charaters operate correctly."""
        self.assertEqual(titlecase("a_b", delim="-"), "A-B")

    def test_capitalize_first_word(self):
        """Check capitalize_first operates correctly."""
        self.assertEqual(titlecase("a_b", capitalize_first=True), "AB")
        self.assertEqual(titlecase("a_b", capitalize_first=False), "aB")

    def test_exceptions(self):
        """Check execptions operate correctly."""
        self.assertEqual(titlecase("Sample_AAA", exceptions=["AAA"]), "SampleAAA")
        self.assertEqual(titlecase("Sample_aaa", exceptions=["aaa"]), "Sampleaaa")

    def test_abbreviations(self):
        """Check abbreviations operate correctly."""
        self.assertEqual(titlecase("Sample_ID", abbrv=["ID"]), "SampleID")

    def test_original_camelcase(self):
        """Check whether original camelcase is preserved."""
        pass


class TestParseEntry(unittest.TestCase):
    """Tests the regex parser for data munging string --> value conversion."""

    def setUp(self):
        self.regex = r"(\s)*?(?P<value>[\w]+)(\s)*?"

    def test_single_entry(self):
        for value in ["A", " A ", " A  ", " A _1", "A .[1]"]:
            with self.subTest(value=value):
                parsed = parse_entry(value, regex=self.regex)
                self.assertEqual(parsed, "A")

    def test_multiple_value_return(self):
        delimiter = ","
        for value in ["A, B", " A ,B ", " A , B", " A ,B _", "A, B.[1]"]:
            with self.subTest(value=value):
                parsed = parse_entry(
                    value, regex=self.regex, delimiter=",", first_only=False
                )
                self.assertEqual(parsed, ["A", "B"])

    def test_first_only(self):
        delimiter = ","
        for value, fo, expect in [("A, B", True, "A"), ("A, B", False, ["A", "B"])]:
            with self.subTest(value=value, fo=fo, expect=expect):
                parsed = parse_entry(
                    value, regex=self.regex, delimiter=",", first_only=fo
                )
                self.assertEqual(parsed, expect)

    def test_delimiters(self):
        for value, delim in [("A, B", ","), ("A; B", ";"), ("A- B", "-"), ("A B", " ")]:
            with self.subTest(value=value, delim=delim):
                parsed = parse_entry(
                    value, regex=self.regex, delimiter=delim, first_only=False
                )
                self.assertEqual(parsed, ["A", "B"])

    def test_replace_nan(self):
        for null_value in [np.nan, None]:
            for n in [np.nan, "None", None]:
                with self.subTest(n=n, null_value=null_value):
                    parsed = parse_entry(
                        null_value, regex=self.regex, delimiter=",", replace_nan=n
                    )
                    if n is None:
                        self.assertTrue(parsed is None)
                    elif isinstance(n, str):
                        self.assertTrue(parsed == n)
                    else:
                        self.assertTrue(np.isnan(parsed))

    def test_values_only(self):
        regex = r"(\s)*?(?P<value>[\w]+)(\s)*?"
        for value, vo, expect in [
            ("A, B", True, ["A", "B"]),
            ("A, B", False, [{"value": "A"}, {"value": "B"}]),
        ]:
            with self.subTest(value=value):
                parsed = parse_entry(
                    value, regex=regex, delimiter=",", first_only=False, values_only=vo
                )
                self.assertEqual(parsed, expect)


class TestStringVariations(unittest.TestCase):
    def setUp(self):
        self.single = "single"
        self.multiple = ["Multiple-a", "multiple-b"]

    def test_single(self):
        self.assertEqual(
            string_variations(self.single), string_variations([self.single])
        )

    def test_preprocess(self):
        for ps in [["lower"], ["upper"], ["upper", "lower"]]:
            string_variations(self.single, preprocess=ps)

    def test_multiple(self):
        string_variations(self.multiple)


class TestSplitRecords(unittest.TestCase):
    """Tests the regex parser for poorly formatted records."""

    def test_split_csv_rows(self):
        self.assertEqual(split_records("a,'b'\r\nc,'d'"), ["a,'b'", "c,'d'"])


class TestSlugify(unittest.TestCase):
    def test_default(self):
        self.assertEqual(slugify("a b"), "a-b")
        self.assertEqual(slugify("a@b"), "ab")
        self.assertEqual(slugify(r"a%%b"), "ab")
        self.assertEqual(slugify(r"a%% b"), "a-b")


class TestInt2Alpha(unittest.TestCase):
    def test_default(self):
        self.assertEqual(int_to_alpha(0), "a")
        self.assertEqual(int_to_alpha(26), "aa")


if __name__ == "__main__":
    unittest.main()
