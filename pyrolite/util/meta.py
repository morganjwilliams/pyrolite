import importlib
import inspect
import webbrowser
from pathlib import Path

from .log import Handle

logger = Handle(__name__)


def get_module_datafolder(module="pyrolite", subfolder=None):
    """
    Returns the path of a module data folder.

    Parameters
    -----------
    subfolder : :class:`str`
        Subfolder within the module data folder.

    Returns
    -------
    :class:`pathlib.Path`
    """
    pth = Path(importlib.util.find_spec(module).origin).parent / "data"
    if subfolder:
        pth /= subfolder
    return pth


def pyrolite_datafolder(subfolder=None):
    """
    Returns the path of the pyrolite data folder.

    Parameters
    -----------
    subfolder : :class:`str`
        Subfolder within the pyrolite data folder.

    Returns
    -------
    :class:`pathlib.Path`
    """
    return get_module_datafolder(module="pyrolite", subfolder=subfolder)


def take_me_to_the_docs():
    """Opens the pyrolite documentation in a webbrowser."""
    webbrowser.open("https://pyrolite.rtfd.io")


def sphinx_doi_link(doi):
    """
    Generate a string with a restructured text link to a given DOI.

    Parameters
    ----------
    doi : :class:`str`

    Returns
    --------
    :class:`str`
        String with doi link.
    """
    return f"`{doi} <https://dx.doi.org/{doi}>`__"


def subkwargs(kwargs, *f):
    """
    Get a subset of keyword arguments which are accepted by a function.

    Parameters
    ----------
    kwargs : :class:`dict`
        Dictionary of keyword arguments.
    f : :class:`callable`
        Function(s) to check.

    Returns
    --------
    :class:`dict`
        Dictionary containing only relevant keyword arguments.
    """
    return {k: v for k, v in kwargs.items() if inargs(k, *f)}


def inargs(name, *funcs):
    """
    Check if an argument is a possible input for a specific function.

    Parameters
    ----------
    name : :class:`str`
        Argument name.
    f : :class:`callable`
        Function(s) to check.

    Returns
    --------
    :class:`bool`
    """
    args = []
    for f in funcs:
        args += list(inspect.signature(f).parameters)
    return name in set(args)


def update_docstring_references(obj, ref="ref"):
    """
    Updates docstring reference names to strings including the function name.
    Decorator will return the same function with a modified docstring. Sphinx
    likes unique names - specifically for citations, not so much for footnotes.

    Parameters
    -----------
    obj : :class:`func` | :class:`class`
        Class or function for which to update documentation references.
    ref : :class:`str`
        String to replace with the object name.

    Returns
    -------
    :class:`func` | :class:`class`
        Object with modified docstring.
    """
    name = obj.__name__
    if hasattr(obj, "__module__"):
        name = obj.__module__ + "." + name
    obj.__doc__ = str(obj.__doc__).replace(ref, name)
    return obj
