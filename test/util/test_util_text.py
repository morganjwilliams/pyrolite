import unittest

from pyrolite.util.text import quoted_string, titlecase
import numpy as np

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

if __name__ == '__main__':
    unittest.main()
