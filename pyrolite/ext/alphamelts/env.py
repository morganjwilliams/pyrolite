"""
alphaMELTS environment managment.
"""
import os
import logging
from pyrolite.util.env import validate_update_envvar
from pyrolite.util.text import remove_prefix
from textwrap import dedent
from ...data.alphamelts.env import MELTS_environment_variables

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def output_formatter(value):
    """
    Output formatter for environment variable values.

    Parameters
    ------------
    value
        Value to format.

    Returns
    --------
    :class:`str`
        Formatted value.
    """
    if value is not None and not isinstance(value, bool):
        return str(value)
    else:
        return ""


class MELTS_Env(object):
    """
    Melts environment object.

    Todo
    -----
        * Implement use as context manager.
    """

    def __init__(self, prefix="ALPHAMELTS_", variable_model=None):
        self.prefix = prefix
        if variable_model is None:
            variable_model = MELTS_environment_variables
        self.spec = variable_model
        self.force_active = False
        self.output_formatter = output_formatter
        self.export_default_env()

    def export_default_env(self):
        """
        Parse any environment variables which are already set.
        Reset environment variables after substituting defaults for unset
        variables.
        """
        _dump = self.dump(prefix=False)

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

    def dump(self, unset_variables=True, prefix=False, cast=lambda x: x):
        r"""
        Export environment configuration to a dictionary.

        Parameters
        -----------
        unset_variables : :class:`bool`
            Whether to include variables which are currently unset.
        prefix : :class:`bool`
            Whether to prefix environment variables (i.e with ALPHAMELTS\_).
        cast : :class:`callable`
            Function to cast environment variable values.

        Returns
        --------
        :class:`dict`
            Dictionary of environent variables and their values.
        """
        keys = [k for k in self.spec.keys()]
        pkeys = [self.prefix + k for k in keys]
        values = [os.getenv(p) for p in pkeys]
        types = [
            self.spec[k]["type"] if self.spec[k].get("type", None) is not None else str
            for k in keys
        ]

        # Evironment variable are always imported as strings
        _env = [
            (k, t(v)) if v and (v not in [None, "None"]) else (k, None)
            for k, p, v, t in zip(keys, pkeys, values, types)
        ]
        if not unset_variables:
            _env = [e for e in _env if e[1] is not None]
        return {[k, self.prefix + k][prefix]: cast(v) for k, v in _env}

    def to_envfile(self, unset_variables=False):
        """
        Create a string representation equivalent to the alphamelts defualt
        environment file.

        Parameters
        -----------
        unset_variables : :class:`bool`
            Whether to include unset variables in the output (commented out).

        Returns
        -------
        :class:`str`
            String-representation of the environment which can be writen to a file.
        """
        preamble = dedent(
            """
        ! Default values of environment variables (pyrolite export)
        ! Variables preceeded by '!' are 'unset' (i.e. 'false')
        """
        )
        return preamble + "\n".join(
            [
                ["", "!"][v is None] + "{} {}".format(k, v)
                for k, v in self.dump(
                    prefix=True, unset_variables=unset_variables
                ).items()
            ]
        )

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
