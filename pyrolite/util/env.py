import os, sys
from contextlib import contextmanager
from .general import iscollection
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


@contextmanager
def environment_manager(env):
    """
    Temporarily set environment variables inside the context manager and
    fully restore previous environment afterwards.
    """
    original_env = {key: os.getenv(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


def validate_value(value, validator):
    """Validates a value based on one or a series of validator functions."""

    if iscollection(validator):
        return all([f(value) for f in validator if callable(f)])
    else:
        return validator(value)


def validate_update_envvar(
    key, value=None, prefix="", force_active=False, variable_model={}, formatter=str
):
    """
    Updates an environment variable after validation.

    parameters

    """
    schema = variable_model.get(key, None)
    if schema is not None:  # some potential validation
        if value is not None:
            if schema.get("validator", None) is not None:
                valid = validate_value(value, schema["validator"])
                assert valid, "Invalid value for parameter {}: {}".format(var, value)

            if schema.get("overridden_by", None) is not None:
                # check for overriders
                overriders = [prefix + k for k in schema.get("overridden_by")]

                if any([over in os.environ for over in overriders]):
                    if force_active:
                        for k in [key for key in overriders if key in os.environ]:
                            del os.environ[key]
        else:
            # try to set to default
            if schema.get("default", None) is not None:
                if schema.get("dependent_on", None) is None:
                    default = schema["default"]
                else:
                    conditions = {
                        k: variable_model[k]["default"]
                        for k in schema["dependent_on"]
                        if variable_model[k]["default"] is not None
                    }
                    default = schema["default"](conditions)
                value = default

    if value is not None:
        logging.debug("EnvVar {} set to {}.".format(prefix + key, value))
        os.environ[prefix + key] = formatter(value)
    else:  # Remove the environment variable if it exists
        if prefix + key in os.environ:
            logging.debug("EnvVar {} removed.".format(prefix + key))
            del os.environ[prefix + key]
