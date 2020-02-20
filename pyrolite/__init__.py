"""
pyrolite: A set of tools for getting the most from your geochemical data.
"""
import sys
import logging
import importlib
import pkgutil

# http://docs.python-guide.org/en/latest/writing/logging/
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.captureWarnings(True)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def load_extensions(base="pyrolite_", replace=["util"]):
    """
    Automatically load any extensions associated with pyrolite
    to be importable from :mod:`pyrolite.extensions`.

    Parameters
    ------------
    base : :class:`str`
        Module base string pattern for recognising extensions.
    replace : :class:`list`
        List of strings to replace from extension modules to shorten call signatures.
    """
    from . import extensions

    modules = {
        name.replace(base, ""): importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith(base)
    }
    output = {}
    for n, m in modules.items():
        for r in replace:
            n = n.replace(r, "")
        setattr(extensions, n, m)
