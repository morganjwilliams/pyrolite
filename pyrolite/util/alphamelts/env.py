import os
import logging
from pyrolite.util.env import validate_update_envvar
from pyrolite.util.text import remove_prefix
from pyrolite.data.alphamelts.env import MELTS_environment_variables

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def output_formatter(value):
    """Output formatter for environment variable values."""
    if value and (value is not None):
        return str(value)
    else:
        return ""


class MELTS_Env(object):
    def __init__(
        self, prefix="ALPHAMELTS_", variable_model=MELTS_environment_variables
    ):
        self.prefix = prefix
        self.spec = variable_model
        self.force_active = False
        self.output_formatter = output_formatter
        self.export_default_env(init=True)

    def export_default_env(self, init=False):
        """
        Parse any environment variables which are already set.
        Reset environment variables after substituding defaults for unset
        variables.
        """
        _dump = self.dump()

        for var, template in self.spec.items():
            _spec = template
            name = self.prefix + var
            is_already_set = (name in os.environ) or (var in _dump.keys())
            if not is_already_set and _spec["set"]:
                setting = True  # should be set by default
            elif is_already_set and _spec["set"]:
                setting = True  # set, should be set by default
            elif is_already_set and not _spec["set"]:
                setting = True  # set, not set by default
            elif not is_already_set and not _spec["set"]:
                setting = False  # not set, should not be set

            if setting:
                setattr(self, var, None)

    def dump(self, unset_variables=True):
        """Export environment configuration to a dictionary."""
        keys = [k for k in self.spec.keys()]
        pkeys = [self.prefix + k for k in keys]
        values = [os.getenv(p) for p in pkeys]
        types = [
            self.spec[k]["type"] if self.spec[k].get("type", None) is not None else str
            for k in keys
        ]

        # Evironment variable are always imported as strings
        _env = [
            (k, t(v)) if v and v not in [None, "None"] else (k, None)
            for k, p, v, t in zip(keys, pkeys, values, types)
        ]
        if not unset_variables:
            _env = [e for e in _env if e[1] is not None]
        return {k: v for k, v in _env}

    def __setattr__(self, name, value):
        """
        Custom setattr to set environment variables.

        Setting attributes with or without the specified prefix should set
        the appropriate prefixed environment variable.
        """

        if hasattr(self, "spec"):
            prefix = getattr(self, "prefix", "")
            dump = self.dump()
            name = remove_prefix(name, prefix)
            if name in self.spec:
                validate_update_envvar(
                    name,
                    value=value,
                    prefix=self.prefix,
                    force_active=self.force_active,
                    variable_model=self.spec,
                    formatter=self.output_formatter,
                )
            else:  # other object attributes
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    def __repr__(self):
        """Returns the class and all set variables."""
        return "{}({})".format(
            self.__class__.__name__, self.dump(unset_variables=False)
        ).replace(",", ",\n\t\t")
