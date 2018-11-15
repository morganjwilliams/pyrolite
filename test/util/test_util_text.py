import unittest
import numpy as np
from pyrolite.util.text import quoted_string, titlecase, to_width, \
                               remove_prefix, normalise_whitespace, \
                               string_variations

class TestRemovePrefix(unittest.TestCase):

    def test_prefix_present(self):
        pass

    def test_prefix_notpresent(self):
        #
        #for prefix in ['A_A_', 'B_', 'C_']:
        #    with

        #self.assertFalse(s.startswith(prefix))
        pass

    def test_double_prefix(self):
        """Should just remove one prefix."""

        pass


class TestNormaliseWhitespace(unittest.TestCase):

    def test_whitepace_removal(self):


        pass

    def test_whitespace_preservation(self):

        pass


class TestToWidth(unittest.TestCase):

    def test_width_spec(self):
        """
        Check that the output width is as specifed.
        For strings including whitespace, it should
        preserve word structure.
        """

        s = "*- "*100
        for width in [1, 10, 79]:
            with self.subTest(width=width):
                out = to_width(s, width=width)
                w = len(out.splitlines()[0])
                self.assertTrue(w <= width)
                self.assertTrue(w >= width-len(s))


class TestQuotedString(unittest.TestCase):
    """Tests for quoted string utility."""

    def test_quoted_string(self):
        """Check that quoted strings operate correctly."""

        for string in ['singlephrase', 'double phrase', 'tri ple phrase',
                        "singlephrase", "double phrase", "tri ple phrase",
                        """singlephrase""", """double phrase""",
                        """tri ple phrase"""]:

            with self.subTest(string=string):

                quotes = ["'", '"']
                stringcontent = ''.join(i for i in quoted_string(string)
                                        if i not in quotes)
                # Check same content
                self.assertEqual(stringcontent, ''.join(i for i in string))

                contains = [[s in q for s in quoted_string(string)]
                            for q in quotes]
                # Check there's at least two of the same quotes.
                self.assertTrue((np.array(contains).sum()>=2).any())


class TestTitlecase(unittest.TestCase):
    """Tests for titlecase string utility."""

    def test_single_word(self):
        """Check single word cases operate correctly."""
        for string in ['lowercase', 'UPPERCASE', 'CamelCase',
                       'Titlecase']:
            with self.subTest(string=string):
                # String content doesnt change
                self.assertEqual(titlecase(string).lower(),  string.lower())
                # Single word capitalisation
                self.assertEqual(titlecase(string), string[0].upper() + \
                                                    string[1:].lower())

    def test_multi_word(self):
        """Check multiword case operates correctly."""
        for string in ['lower case', 'UPPER CASE', 'Camel Case',
                       'Title case']:
            with self.subTest(string=string):
                # String content doesnt change
                self.assertEqual(titlecase(string).lower(),
                                string.lower().replace(" ", ""))

                expected =  ''.join([st[0].upper() + st[1:].lower()
                                     for st in string.split(' ')])
                self.assertEqual(titlecase(string), expected)

    def test_valid_split_delimiters(self):
        """Check split delimiters operate correctly."""
        for string in ['lower case', 'lower-case', 'lower_case',
                       'lower\tcase', 'lower\ncase']:
            with self.subTest(string=string):
                self.assertEqual(titlecase(string).lower(), 'lowercase')

    @unittest.expectedFailure
    def test_invalid_split_delimiters(self):
        """Check split delimiters operate correctly."""
        for string in ['lower/case', 'lower\case', 'lower&case']:
            with self.subTest(string=string):
                self.assertEqual(titlecase(string).lower(), 'lowercase')

    def test_initial_space(self):
        """Check that strings are stripped effectively."""
        pass

    def test_join_characters(self):
        """Check join charaters operate correctly."""
        pass

    def test_capitalize_first_word(self):
        """Check capitalize_first operates correctly."""
        pass

    def test_exceptions(self):
        """Check execptions operate correctly."""
        pass

    def test_abbreviations(self):
        """Check abbreviations operate correctly."""
        pass

    def test_original_camelcase(self):
        """Check whether original camelcase is preserved."""


class TestParseEntry(unittest.TestCase):
    """Tests the regex parser for data munging string --> value conversion."""

    def setUp(self):
        pass

    def test_single_entry(self):
        pass

    def test_multiple_groups(self):
        pass

    def test_multiple_entries(self):
        pass

    def test_delimiters(self):
        pass

    def test_error_values(self):
        pass

    def test_values_only(self):
        pass


class TestStringVariations(unittest.TestCase):

    def setUp(self):
        self.single = 'single'
        self.multiple = ['Multiple-a', 'multiple-b']

    def test_single(self):
        self.assertEqual(string_variations(self.single),
                         string_variations([self.single]))

    def test_preprocess(self):
        for ps in [['lower'], ['upper'], ['upper', 'lower']]:
            string_variations(self.single, preprocess=ps)

    def test_multiple(self):
        string_variations(self.multiple)



class TestSplitRecords(unittest.TestCase):
    """Tests the regex parser for poorly formatted records."""

    def setUp(self):
        pass

    def test_single_entry(self):
        pass

    def test_delimiters(self):
        pass


if __name__ == '__main__':
    unittest.main()
