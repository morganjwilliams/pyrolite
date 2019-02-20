import pandas_flavor as pf
import pandas as pd
import re
from ..util.text import titlecase
from .validate import is_isotoperatio
from .ind import (
    __common_oxides__,
    __common_elements__,
    get_cations,
    common_elements,
    common_oxides,
    get_isotopes
)
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def tochem(strings: list, abbrv=["ID", "IGSN"], split_on="[\s_]+"):
    """
    Converts a list of strings containing come chemical compounds to
    appropriate case.

    Parameters
    ----------
    strings : :class:`list`
        Strings to convert to 'chemical case'.
    abbr : :class:`list`, ['ID', 'IGSN']
        Abbreivated phrases to ignore in capitalisation.
    split_on : :class:`str`, "[\s_]+"
        Regex for character or phrases to split the strings on.

    Returns
    --------
    :class:`list`
    """
    # accomodate single string passed
    if not type(strings) in [list, pd.core.indexes.base.Index]:
        strings = [strings]

    # translate elements and oxides
    chems = __common_elements__ | __common_oxides__
    trans = {str(e).upper(): str(e) for e in chems}
    strings = [trans[str(h).upper()] if str(h).upper() in trans else h for h in strings]

    # translate potential isotope ratios
    strings = [repr_isotope_ratio(h) for h in strings]
    return strings


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
    return "{}{}{}{}".format(num_iso, titlecase(num_el), den_iso, titlecase(den_el))


@pf.register_series_method
@pf.register_dataframe_method
def check_multiple_cation_inclusion(df, exclude=["LOI", "FeOT", "Fe2O3T"]):
    """
    Returns cations which are present in both oxide and elemental form.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe to check duplication within.
    exclude : :class:`list`, ['LOI','FeOT', 'Fe2O3T']
        List of components to exclude from the duplication check.

    Returns
    --------
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
