import logging
import re
from functools import partial
import pandas as pd
from ....util.text import parse_entry, split_records, titlecase
from ....util.types import iscollection
from ....geochem.parse import tochem

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__value_rx__ = r"(\s)*?(?P<value>[\.,\s\w]+\b)((\s)*?\[)?(?P<key>\w*)(\])?(\s)*?"
__cit_rx__ = r"(\s)*?(\[)?(?P<key>\w*)(\])?(\s)*?(?P<value>[\.\w]+)(\s)*?"
__full_cit_rx__ = r"(\s)*?\[(?P<key>\w*)\](\s)*(?P<value>.+)$"
__doi_rx__ = r"(.)*(doi(\s)*?:*)(\s)*(?P<value>\S*)"


def columns_to_namesunits(names):
    _units = [re.findall("\((.+)\)", n) for n in names]
    _units = [i[0].lower().replace(".", "") if i else None for i in _units]
    _unitless = [re.sub("\(.+\)", "", n) for n in names]
    _tnames = [titlecase(n, abbrv=["ID"]) for n in _unitless]  # titlecase/camelcase
    _chemnames = [tochem(n) for n in _tnames]
    return _chemnames, _units


def subsitute_commas(entry):
    if iscollection(entry):
        return [x.replace(",", ";") for x in entry]
    else:
        return entry.replace(",", ";")


def parse_values(entry, sub=subsitute_commas, **kwargs):
    """
    Wrapper for parse_entry for GEOROC formatted values.

    Parameters
    -------------
    entry: pd.Series | str
        String series formated as sequences of 'VALUE [NUMERIC_CITATION]'
        separated by '/'. Else a string entry itself.
    sub: function
        Secondary subsitution function, here used for subsitution
        (e.g. of commas).
    """
    f = partial(parse_entry, regex=__value_rx__, delimiter="/", **kwargs)
    if isinstance(entry, pd.Series):
        return entry.apply(f).apply(sub)
    else:
        return sub(f(entry))


def parse_citations(entry, **kwargs):
    """
    Wrapper for parse_entry for GEOROC formatted citations.

    Parameters
    -------------
    ser: pd.Series
        String series formated as sequences of '[NUMERIC_CITATION] Citation'.
    """
    f = partial(
        parse_entry, regex=__full_cit_rx__, values_only=False, delimiter=None, **kwargs
    )
    if isinstance(entry, pd.Series):
        return entry.apply(f)
    else:
        return f(entry)


def parse_DOI(entry, link=True, **kwargs):
    """
    Wrapper for parse_entry for GEOROC formatted dois.

    Parameters
    -------------
    ser: pd.Series
        String series formated as sequences of 'Citation doi: DOI'.
    """
    f = partial(
        parse_entry,
        regex=__doi_rx__,
        values_only=True,
        delimiter=None,
        first_only=True,
        replace_nan="",
        **kwargs
    )
    if isinstance(entry, pd.Series):
        return entry.apply(lambda x: r"{}{}".format(["", "dx.doi.org/"][link], f(x)))
    else:
        return r"{}{}".format(["", "dx.doi.org/"][link], f(entry))
