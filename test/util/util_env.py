import os, sys
import unittest
from pyrolite.util.env import validate_value, validate_update_envvar


class TestValidateValue(unittest.TestCase):
    def setUp(self):
        self.value = 10

    def test_single_validator(self):
        v = lambda x: x > 0
        expect = True
        self.assertTrue(validate_value(self.value, v) is expect)

    def test_multiple_validator(self):
        for vs, expect in [
            ([lambda x: x > 0, lambda x: x < 20], True),
            ((lambda x: x > 0, lambda x: x < 5), False),
        ]:
            with self.subTest(vs=vs, expect=expect):
                self.assertTrue(validate_value(self.value, vs) is expect)


class TestValidateUpdateEnvvar(unittest.TestCase):
    # validate_update_envvar(key, value=None, prefix="", force_active=False, variable_model=None, formatter=str)
    def setUp(self):
        self.value = 10
        self.default = 20
        self.prefix = "TestValidateUpdateEnvvar_"
        self.schema = {  # Solution Models
            "VALUE": dict(default=self.default, validator=lambda x: x > 6,)
        }

    def test_default(self):
        # should update the value
        validate_update_envvar("VALUE", self.value, prefix=self.prefix)
        self.assertEqual(os.environ[self.prefix + "VALUE"], str(self.value))

    def test_schema_nonnullvalue(self):
        # should update the value where conditions pass
        validate_update_envvar(
            "VALUE", self.value, variable_model=self.schema, prefix=self.prefix
        )
        self.assertEqual(os.environ[self.prefix + "VALUE"], str(self.value))

    @unittest.expectedFailure
    def test_invalid(self):
        # should raise an assertion error
        validate_update_envvar(
            "VALUE", 5, variable_model=self.schema, prefix=self.prefix
        )

    def test_schema_defaultvalue(self):
        # should re-set the value to the default
        validate_update_envvar(
            "VALUE", None, variable_model=self.schema, prefix=self.prefix
        )
        self.assertEqual(os.environ[self.prefix + "VALUE"], str(self.default))

    def test_set_and_remove(self):
        validate_update_envvar(
            "VALUE", 52, variable_model=self.schema, prefix=self.prefix
        )
        validate_update_envvar(
            "VALUE", None, variable_model=self.schema, prefix=self.prefix
        )
        self.assertEqual(os.environ[self.prefix + "VALUE"], str(self.default))

    def test_overridden(self):
        # test yet to be implemented
        pass


if __name__ == "__main__":
    unittest.main()
