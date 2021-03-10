import os
from .types import iscollection
from .log import Handle

logger = Handle(__name__)


def validate_value(value, validator):
    """Validates a value based on one or a series of validator functions."""
    if iscollection(validator):
        return all([f(value) for f in validator if callable(f)])
    else:
        return validator(value)


def validate_update_envvar(
    key, value=None, prefix="", force_active=False, variable_model=None, formatter=str
):
    """
    Updates an environment variable after validation.

    Parameters
    -----------
    key : :class:`str`
        Environment variable name.
    value : :class:`str`
        Value for the environemnt variable.
    force_active : :class:`bool`
        Enforce the schema overriding parameters.
    variable_model : :class:`dict`
        Model of variables indexed by name.
    formatter
        Function for formatting environment variable values.

    """
    variable_model = variable_model or {}  # default value
    schema = variable_model.get(key, None)
    if schema is not None:  # some potential validation
        if value is not None:
            if schema.get("validator", None) is not None:
                valid = validate_value(value, schema["validator"])
                assert valid, "Invalid value for parameter {}: {}".format(key, value)

            if schema.get("overridden_by", None) is not None:
                # check for overriders
                overriders = [prefix + k for k in schema.get("overridden_by")]

                if any([over in os.environ for over in overriders]):
                    # if there are over-riding parameters, remove this one
                    if force_active and any([key in os.environ for key in overriders]):
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
        logger.debug("EnvVar {} set to {}.".format(prefix + key, value))
        os.environ[prefix + key] = formatter(value)
    else:  # Remove the environment variable if it exists
        if prefix + key in os.environ:
            logger.debug("EnvVar {} removed.".format(prefix + key))
            del os.environ[prefix + key]
