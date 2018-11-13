import os, sys
import unittest
from pyrolite.util.env import *

class TestEnvironmentManager(unittest.TestCase):

    def setUp(self):
        self.env = {'test_envar': 'test_env_value'}

    def test_env_manager_context(self):

        with environment_manager(self.env) as env:
            pass

    def tearDown(self):
        for k, v in self.env.items():
            if os.getenv(k) is not None:
                del os.environ[k]


# validate_value


# validate_update_envvar


if __name__ == '__main__':
    unittest.main()
