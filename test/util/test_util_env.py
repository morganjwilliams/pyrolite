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


class TestValidateValue(unittest.TestCase):

    def setUp(self):
        self.value = 10

    def test_single_validator(self):
        v = lambda x: x>0
        expect = True
        self.assertTrue(validate_value(self.value, v) is expect)

    def test_multiple_validator(self):
        for vs, expect in [([lambda x: x>0, lambda x: x<20], True),
                           ((lambda x: x>0, lambda x: x<5), False),
                           ]:
            with self.subTest(vs=vs, expect=expect):
                self.assertTrue(validate_value(self.value, vs) is expect)



if __name__ == '__main__':
    unittest.main()
