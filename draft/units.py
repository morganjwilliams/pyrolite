import unittest
import pandas as pd


def patch_pandas_units():
    """
    Patches pandas dataframes and series to have units values.
    Todo: implement auto-assign of pandas units at __init__
    """

    def set_units(df: pd.DataFrame, units):
        """
        Monkey patch to add units to a dataframe.
        """

        if isinstance(df, pd.DataFrame):
            if isinstance(units, str):
                units = np.array([units])
            units = pd.Series(units, name="units")
        elif isinstance(df, pd.Series):
            assert isinstance(units, str)
            pass
        setattr(df, "units", units)

    """
    def init_wrapper(func, *args, **kwargs):

        def init_wrapped(*args, **kwargs):
            func(*args, **kwargs)

        units = kwargs.pop('units', None)
        if units is not None:
            if isinstance(args[0], pd.DataFrame):
                df = args[0]
                df.set_units(units)
        return init_wrapped
    """

    for cls in [pd.DataFrame, pd.Series]:
        setattr(cls, "units", None)
        setattr(cls, set_units.__name__, set_units)
        # setattr(cls, '__init__', init_wrapper(cls.__init__))


class TestPandasUnitsPatch(unittest.TestCase):
    @unittest.expectedFailure
    def test_classes(self):
        """
        Check patching hasn't occurred to start with.
        """
        for cls in [pd.DataFrame, pd.Series]:
            with self.subTest(cls=cls):
                for prop in [units, set_units]:  # units property  # set_units method
                    with self.subTest(prop=prop):
                        clsp = getattr(cls, prop)
                        instp = getattr(cls(), prop)

    def test_patch(self):
        """
        Check that the units attribute exists after patching.
        """
        patch_pandas_units()
        for cls in [pd.DataFrame, pd.Series]:
            with self.subTest(cls=cls):
                for prop in [
                    "units",  # units property
                    "set_units",  # set_units method
                ]:
                    with self.subTest(prop=prop):
                        clsp = getattr(cls, prop)
                        instp = getattr(cls(), prop)

    def test_set_units(self):
        """
        Check that the set_units method works after patching.
        """
        patch_pandas_units()
        test_units = "Wt%"
        for cls in [pd.DataFrame, pd.Series]:
            with self.subTest(cls=cls):
                instance = cls()
                instance.set_units("Wt%")
                equiv = instance.units == test_units
                if not isinstance(equiv, bool):
                    equiv = equiv.all()
                    self.assertTrue(isinstance(instance.units, pd.Series))

                self.assertTrue(equiv)
