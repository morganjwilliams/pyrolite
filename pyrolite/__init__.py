"""
pyrolite: A set of tools for getting the most from your geochemical data.
"""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from pathlib import Path
import importlib
import pkgutil
import matplotlib.style
from .plot import _export_mplstyle
from .util.log import Handle

logger = Handle(__name__)


def load_extensions(base="pyrolite_", replace=["util"]):
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
