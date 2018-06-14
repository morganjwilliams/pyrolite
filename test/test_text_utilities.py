import unittest

from pyrolite.text_utilities import quoted_string, titlecase

class TestQuotedString(unittest.TestCase):
    """Tests for quoted string utility."""

    def test_quoted_string(self):
        """Check that quoted strings operate correctly."""

        for string in ['singlephrase', 'double phrase', 'tri ple phrase',
                        "singlephrase", "double phrase", "tri ple phrase",
                        """singlephrase""", """double phrase""",
                        """tri ple phrase"""]:
            quoted_string(string)


class TestTitlecase(unittest.TestCase):
    """Tests for titlecase string utility."""

    def test_single_word(self):
        """Check single word cases operate correctly."""
        pass

    def test_multi_word(self):
        """Check multiword case operates correctly."""
        pass

    def test_split_delimiters(self):
        """Check split delimiters operate correctly."""
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
