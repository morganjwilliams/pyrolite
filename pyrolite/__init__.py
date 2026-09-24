"""
pyrolite: A set of tools for getting the most from your geochemical data.
"""

import importlib
import pkgutil

import matplotlib.style

from ._version import __version__

# initialise pandas accessors
from .comp import pyrocomp  # noqa: F401
from .geochem import pyrochem  # noqa: F401
from .plot import pyroplot  # noqa: F401
from .util.log import Handle

logger = Handle(__name__)

__all__ = ["Handle", "__version__", "load_extensions"]


def load_extensions(base="pyrolite_", replace=None):
    """
    Automatically load any extensions associated with pyrolite
    to be importable from :mod:`pyrolite.extensions`.

    Parameters
    ----------
    base : :class:`str`
        Module base string pattern for recognising extensions.
    replace : :class:`list`
        List of strings to replace from extension modules to shorten call signatures.
    """
    from . import extensions

    if replace is None:
        replace = ["util"]
    modules = {
        name.replace(base, ""): importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith(base)
    }
    for n, m in modules.items():
        for r in replace:
            n = n.replace(r, "")
        setattr(extensions, n, m)


# _export_pyrolite_mplstyle() should be called in .plot import regardless
matplotlib.style.use("pyrolite")
