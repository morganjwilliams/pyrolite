import unittest
import pandas as pd
import pyrolite.util.time as pyrotime


class TestTimescale(unittest.TestCase):
    """
    Test the construction of the Timescale utility class.
    """

    def setUp(self):
        self.ts = pyrotime.Timescale()
        self.test_ages = [0., 10., 1000., 4100.]

    def test_default_properties(self):
        """
        Test that the default groups are present, and can be accessed
        as pd.DataFrames.
        """

        for g in ['Eons', 'Eras', 'Periods', 'Ages']:
            with self.subTest(g=g):
                grp = getattr(self.ts, g)
                self.assertIsInstance(grp, pd.DataFrame)

    def test_named_age_int_age(self):
        ages = [int(a) for a in self.test_ages]
        for a in ages:
            with self.subTest(a=a):
                self.ts.named_age(a)

    def test_named_age_float_age(self):
        ages = [float(a) for a in self.test_ages]
        for a in ages:
            with self.subTest(a=a):
                self.ts.named_age(a)

    def test_named_age_levels(self):
        age = self.test_ages[1]
        levels = ['Eon', 'Era', 'Period', 'Age', 'Specific']
        for l in levels:
            with self.subTest(l=l):
                self.ts.named_age(age, level=l)

    @unittest.expectedFailure
    def test_named_age_negatives_age(self):
        ages = [-float(a) for a in self.test_ages]
        for a in ages:
            with self.subTest(a=a):
                self.ts.named_age(a)

    def test_text2age_true_ages(self):
        for l in self.ts.levels:
            ages = self.ts.data.loc[self.ts.data.Level==l, 'Name'].unique()
            for a in ages:
                with self.subTest(a=a):
                    self.ts.text2age(a)

    @unittest.expectedFailure
    def test_text2age_false_ages(self):
        ages = ['not an age', 'unknown', 'None']
        for a in ages:
            with self.subTest(a=a):
                self.ts.text2age(a)


class TestTimescaleReferenceFrame(unittest.TestCase):
    """
    Test that the reference dataframe can be built.
    """

    def setUp(self):
        self.filename = pyrotime.__DATA__

    def test_frame_build(self):
        data = pyrotime.timescale_reference_frame(self.filename)



if __name__ == "__main__":
    unittest.main()
