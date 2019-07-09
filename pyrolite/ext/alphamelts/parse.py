"""
Parsing utilities for use with alphaMELTS.
"""
import re
import pandas as pd
from pathlib import Path
import logging
import periodictable as pt
from ...mineral.mineral import merge_formulae
from .env import MELTS_Env
from .meltsfile import to_meltsfile

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def _file_from_obj(fileobj):
    """
    Read in file data either from a file path or a string.

    Parameters
    ------------
    fileobj : :class:`str` | :class:`pathlib.Path`
        Either a path to a valid file, or a multiline string representation of a
        file object.

    Returns
    --------
    file : :class:`str`
        Multiline string representation of a file.
    path
        Path to the original file, if it exists.

    Notes
    ------
        This function deconvolutes the possible ways in which one can pass either
        a file, or reference to a file.

    Todo
    ----
        * Could be passed an open file object
    """
    path, file = None, None
    if isinstance(fileobj, Path):
        path = str(fileobj)
    elif isinstance(fileobj, str):
        if len(re.split("[\r\n]", fileobj)) > 1:  # multiline string passed as a file
            file = fileobj
        else:  # path passed as a string
            path = fileobj
    else:
        pass

    if (path is not None) and (file is None):
        with open(path) as f:
            file = f.read()

    assert file is not None  # can't not have a meltsfile
    return file, path


def read_meltsfile(meltsfile, **kwargs):
    """
    Read in a melts file from a :class:`~pandas.Series`, :class:`~pathlib.Path` or
    string.

    Parameters
    ------------
    meltsfile : :class:`pandas.Series` | :class:`str` | :class:`pathlib.Path`
        Either a path to a valid melts file, a :class:`pandas.Series`, or a
        multiline string representation of a melts file object.

    Returns
    --------
    file : :class:`str`
        Multiline string representation of a meltsfile.
    path
        Path to the original file, if it exists.

    Notes
    ------
        This function deconvolutes the possible ways in which one can pass either
        a meltsfile, or reference to a meltsfile.
    """
    path, file = None, None
    if isinstance(meltsfile, pd.Series):
        file = to_meltsfile(meltsfile, **kwargs)
    else:
        file, path = _file_from_obj(meltsfile)
    return file, path


def read_envfile(envfile, **kwargs):
    """
    Read in a environment file from a  :class:`~pyrolite.util.alphamelts.env.MELTS_Env`,
    :class:`~pathlib.Path` or string.

    Parameters
    ------------
    envfile : :class:`~pyrolite.util.alphamelts.env.MELTS_Env` | :class:`str` | :class:`pathlib.Path`
        Either a path to a valid environment file, a :class:`pandas.Series`, or a
        multiline string representation of a environment file object.

    Returns
    --------
    file : :class:`str`
        Multiline string representation of an environment file.
    path
        Path to the original file, if it exists.
    """
    path, file = None, None
    if isinstance(envfile, MELTS_Env):
        file = envfile.to_envfile(**kwargs)
    else:
        file, path = _file_from_obj(envfile)
    return file, path


def from_melts_cstr(composition_str, formula=True):
    """
    Parses melts composition strings to dictionaries or formulae.

    Parameters
    -----------
    composition_str : :class:`str`
        Composition to parse.
    formula : :class:`bool`
        Whether to output a :class:`periodictable.formula.Formula`

    Returns
    --------
    :class:`dict` | :class:`periodictable.formulas.Formula`
        Dictionary containing components, or alternatively if :code:`formula = True`,
        a :class:`~periodictable.formulas.Formula` representation of the composition.

    Todo
    ------
        * Enable parsing of nested brackets in composition.
    """
    regex = r"""(?P<el>[a-zA-Z']+)(?P<num>[^a-zA-Z()]+)"""
    sub = r"""[\']+"""

    def repl(m):
        return (
            "{" + str(m.group(0).count("""'""")) + "+" + "}"
        )  # replace ' with count(')

    tfm = lambda s: re.sub(sub, repl, s)
    if not formula:
        result = re.findall(regex, composition_str)
        result = [(tfm(el), float(val)) for el, val in result]
        return {k: v for k, v in result}
    else:
        # entries with rounding errors giving negative 0 won't match
        result = tfm(composition_str.replace("-0.0", "0.0"))
        return pt.formula(result)
