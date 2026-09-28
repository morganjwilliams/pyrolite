"""
Functions for parsing, formatting and validating chemical names and formulae.
"""

import re

import pandas as pd
import periodictable as pt

from ..util.log import Handle
from ..util.text import titlecase
from .ind import (
    _common_elements,
    _common_oxides,
    common_elements,
    get_cations,
    get_isotopes,
)

logger = Handle(__name__)


def is_isotoperatio(
    s: str, require_split: bool = False, split_on: str = r"[\s_]+"
) -> bool:
    """
    Check if text is plausibly an isotope ratio.

    Parameters
    -----------
    s : str
        String to validate.

    Returns
    --------
    bool

    Todo
    -----
        * Validate the isotope masses vs natural isotopes
    """
    if s not in _common_oxides:
        isotopes = get_isotopes(s)
        return len(isotopes) == 2
    else:
        return False


def repr_isotope_ratio(isos: str | tuple[str]) -> str | None:
    """
    Format an isotope ratio pair as a string.

    Parameters
    -----------
    isotope_ratio : tuple
        Numerator, denominator pair.

    Returns
    --------
    str

    Todo
    -----
    Consider returning additional text outside of the match (e.g. 87Sr/86Sri should
    include the 'i').
    """
    isomatch = r"([0-9][0-9]?[0-9]?)"
    elmatch = r"([a-zA-Z][a-zA-Z]?)"
    if not is_isotoperatio(isos):
        return isos
    else:
        if isinstance(isos, str):
            isos: tuple | None = get_isotopes(isos)

        if isos:
            num, den = isos

            num_iso, num_el = re.findall(isomatch, num)[0], re.findall(elmatch, num)[0]
            den_iso, den_el = re.findall(isomatch, den)[0], re.findall(elmatch, den)[0]
            return f"{num_iso}{titlecase(num_el)}/{den_iso}{titlecase(den_el)}"


def ischem(s: str | list[str]) -> bool | list[bool]:
    """
    Checks if a string corresponds to chemical component (compositional).
    Here simply checking whether it is a common element or oxide.

    Parameters
    ----------
    s : str
        String to validate.

    Returns
    -------
    bool

    Todo
    -----
    * Implement checking for other compounds, e.g. carbonates.
    """
    chems = set(map(str.upper, (_common_elements | _common_oxides)))
    if isinstance(s, list):
        return [str(st).upper() in chems for st in s]
    else:
        return str(s).upper() in chems


def tochem(
    strings: str | list[str] | pd.Index,
    abbrv: list[str] | None = None,
    split_on: str = r"[\s_]+",
) -> list[str] | str:
    r"""
    Converts a list of strings containing come chemical compounds to
    appropriate case.

    Parameters
    ----------
    strings : list
        Strings to convert to 'chemical case'.
    abbr : list, `["ID", "IGSN"]`
        Abbreivated phrases to ignore in capitalisation.
    split_on : str, "[\s_]+"
        Regex for character or phrases to split the strings on.

    Returns
    -------
    list | str

    """
    # listify single string passed
    if abbrv is None:
        abbrv = ["ID", "IGSN"]
    listified = False
    if not isinstance(strings, (list, pd.Index)):
        strings = [strings]
        listified = True

    # translate elements and oxides
    # elements second, Co guaranteed to override CO for python 3.6 +

    chems = _common_oxides | _common_elements
    trans = {str(e).upper(): str(e) for e in chems}
    strings = [trans.get(str(h).upper(), h) for h in strings]

    # translate potential isotope ratios
    split_pattern = re.compile(split_on)
    strings = [
        h
        if (h in chems)
        else (repr_isotope_ratio(h) if split_pattern.match(h) is not None else h)
        for h in strings
    ]
    if listified:
        strings = strings[0]
    return strings


def check_multiple_cation_inclusion(
    df: pd.DataFrame | pd.Series, exclude: list[str] | None = None
) -> set[pt.core.Element]:
    """
    Returns cations which are present in both oxide and elemental form.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to check duplication within.
    exclude : list, `["LOI", "FeOT", "Fe2O3T"]`
        List of components to exclude from the duplication check.

    Returns
    -------
    set
        Set of elements for which multiple components exist in the dataframe.

    Todo
    -----
        * Options for output (string/formula).

    """
    if exclude is None:
        exclude = ["LOI", "FeOT", "Fe2O3T"]
    major_components = [i for i in _common_oxides if i in df.columns]
    elements_as_majors = [
        get_cations(oxide)[0] for oxide in major_components if oxide not in exclude
    ]
    elements_as_traces = [
        c for c in common_elements(output="formula") if str(c) in df.columns
    ]
    return {el for el in elements_as_majors if el in elements_as_traces}
