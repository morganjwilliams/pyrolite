import unittest

from pyrolite.text_utilities import quoted_string

class TestTextUtilities(unittest.TestCase):
    """Tests for text utilities"""

    def test_quoted_string(self):
        """Check that quoted strings operate correctly."""

        for string in ['singlephrase', 'double phrase', 'tri ple phrase',
                        "singlephrase", "double phrase", "tri ple phrase",
                        """singlephrase""", """double phrase""",
                        """tri ple phrase"""]:
            quoted_string(string)

if __name__ == '__main__':
    unittest.main()
