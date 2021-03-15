"""
Functions for parsing, formatting and validating chemical names and formulae.
"""
import re
import functools
import pandas as pd
from ..util.text import titlecase
from .ind import (
    __common_oxides__,
    __common_elements__,
    get_cations,
    common_elements,
    get_isotopes,
)
from ..util.log import Handle

logger = Handle(__name__)


def is_isotoperatio(s):
    """
    Check if text is plausibly an isotope ratio.

    Parameters
    -----------
    s : :class:`str`
        String to validate.

    Returns
    --------
    :class:`bool`

    Todo
    -----
        * Validate the isotope masses vs natural isotopes
    """
    if s not in __common_oxides__:
        isotopes = get_isotopes(s)
        return len(isotopes) == 2
    else:
        return False


def repr_isotope_ratio(isotope_ratio):
    """
    Format an isotope ratio pair as a string.

    Parameters
    -----------
    isotope_ratio : :class:`tuple`
        Numerator, denominator pair.

    Returns
    --------
    :class:`str`

    Todo
    -----
    Consider returning additional text outside of the match (e.g. 87Sr/86Sri should
    include the 'i').
    """
    if not is_isotoperatio(isotope_ratio):
        return isotope_ratio
    else:
        if isinstance(isotope_ratio, str):
            isotope_ratio = get_isotopes(isotope_ratio)
        num, den = isotope_ratio
        isomatch = r"([0-9][0-9]?[0-9]?)"
        elmatch = r"([a-zA-Z][a-zA-Z]?)"
        num_iso, num_el = re.findall(isomatch, num)[0], re.findall(elmatch, num)[0]
        den_iso, den_el = re.findall(isomatch, den)[0], re.findall(elmatch, den)[0]
    return "{}{}/{}{}".format(num_iso, titlecase(num_el), den_iso, titlecase(den_el))


def ischem(s):
    """
    Checks if a string corresponds to chemical component (compositional).
    Here simply checking whether it is a common element or oxide.

    Parameters
    ----------
    s : :class:`str`
        String to validate.

    Returns
    -------
    :class:`bool`

    Todo
    -----
        * Implement checking for other compounds, e.g. carbonates.

    """
    chems = set(map(str.upper, (__common_elements__ | __common_oxides__)))
    if isinstance(s, list):
        return [str(st).upper() in chems for st in s]
    else:
        return str(s).upper() in chems


def tochem(strings: list, abbrv=["ID", "IGSN"], split_on=r"[\s_]+"):
    """
    Converts a list of strings containing come chemical compounds to
    appropriate case.

    Parameters
    ----------
    strings : :class:`list`
        Strings to convert to 'chemical case'.
    abbr : :class:`list`, :code:`["ID", "IGSN"]`
        Abbreivated phrases to ignore in capitalisation.
    split_on : :class:`str`, "[\s_]+"
        Regex for character or phrases to split the strings on.

    Returns
    -------
    :class:`list` | :class:`str`

    """
    # listify single string passed
    listified = False
    if not isinstance(strings, (list, pd.core.indexes.base.Index)):
        strings = [strings]
        listified = True

    # translate elements and oxides
    # elements second, Co guaranteed to override CO for python 3.6 +
    chems = __common_oxides__ | __common_elements__
    trans = {str(e).upper(): str(e) for e in chems}
    strings = [trans[str(h).upper()] if str(h).upper() in trans else h for h in strings]

    # translate potential isotope ratios
    strings = [h if (h in chems) else repr_isotope_ratio(h) for h in strings]
    if listified:
        strings = strings[0]
    return strings


def check_multiple_cation_inclusion(df, exclude=["LOI", "FeOT", "Fe2O3T"]):
    """
    Returns cations which are present in both oxide and elemental form.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe to check duplication within.
    exclude : :class:`list`, :code:`["LOI", "FeOT", "Fe2O3T"]`
        List of components to exclude from the duplication check.

    Returns
    -------
    :class:`set`
        Set of elements for which multiple components exist in the dataframe.

    Todo
    -----
        * Options for output (string/formula).

    """
    major_components = [i for i in __common_oxides__ if i in df.columns]
    elements_as_majors = [
        get_cations(oxide)[0] for oxide in major_components if not oxide in exclude
    ]
    elements_as_traces = [
        c for c in common_elements(output="formula") if str(c) in df.columns
    ]
    return set([el for el in elements_as_majors if el in elements_as_traces])
