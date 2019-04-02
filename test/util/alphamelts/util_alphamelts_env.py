import unittest
from pyrolite.data.alphamelts.env import MELTS_environment_variables
from pyrolite.util.text import remove_prefix
from pyrolite.util.alphamelts.env import *


class TestMELTSEnv(unittest.TestCase):
    def setUp(self):
        self.prefix = "ALPHAMELTS_"
        self.env_vars = MELTS_environment_variables

    def test_env_build(self):
        """Tests the environment setup with the default config."""
        menv = MELTS_Env(prefix=self.prefix, variable_model=self.env_vars)
        test_var = "ALPHAMELTS_MINP"
        self.assertTrue(test_var in os.environ)

    def test_valid_setattr(self):
        """Tests that environment variables can be set."""

        menv = MELTS_Env(prefix=self.prefix, variable_model=self.env_vars)
        test_var = "ALPHAMELTS_MINP"
        for var in [test_var, remove_prefix(test_var, self.prefix)]:
            with self.subTest(var=var):
                for value in [1.0, 10.0, 100.0, 10.0]:
                    setattr(menv, var, value)
                    self.assertTrue(test_var in os.environ)
                    self.assertTrue(type(value)(os.environ[test_var]) == value)

    def test_reset(self):
        """
        Tests that environment variables can be reset to default/removed
        by setting to None.
        """
        menv = MELTS_Env(prefix=self.prefix, variable_model=self.env_vars)
        test_var = "ALPHAMELTS_OLD_GARNET"
        for var in [test_var, remove_prefix(test_var, self.prefix)]:
            with self.subTest(var=var):
                setattr(menv, var, True)  # set
                setattr(menv, var, None)  # reset to default/remove
                _var = remove_prefix(var, self.prefix)
                default = self.env_vars[_var].get("default", None)
                if default is not None:
                    self.assertTrue(type(default)(os.environ[test_var]) == default)
                else:
                    self.assertTrue(test_var not in os.environ)


if __name__ == "__main__":
    unittest.main()
