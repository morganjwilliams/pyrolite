import re
import textwrap
import numpy as np
import logging

try:
    from sortedcollections import SortedSet as set
except:
    pass

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()


def to_width(multiline_string, width=79, **kwargs):
    """Uses builtin textwapr for text wrapping to a specific width."""
    return textwrap.fill(multiline_string, width, **kwargs)


def normalise_whitespace(strg):
    """Substitutes extra tabs, newlines etc. for a single space."""
    return re.sub("\s+", " ", strg).strip()


def remove_prefix(z, prefix):
    """Remove a specific prefix from the start of a string."""
    if z.startswith(prefix):
        return re.sub("^{}".format(prefix), "", z)
    else:
        return z


def quoted_string(s):
    # if " " in s or '-' in s or '_' in s:
    s = '''"{}"'''.format(s)
    return s


def titlecase(
    s,
    exceptions=["and", "in", "a"],
    abbrv=["ID", "IGSN", "CIA", "CIW", "PIA", "SAR", "SiTiIndex", "WIP"],
    capitalize_first=True,
    split_on="[\s_-]+",
    delim="",
):
    """
    Formats strings in CamelCase, with exceptions for simple articles
    and omitted abbreviations which retain their capitalization.
    TODO: Option for retaining original CamelCase.
    """
    # Check if abbrv in string, in which case it'll need to be split first?
    words = re.split(split_on, s)
    out = []
    first = words[0]
    if capitalize_first and not (first in abbrv):
        first = first.capitalize()

    out.append(first)
    for word in words[1:]:
        if word in exceptions + abbrv:
            pass
        elif word.upper() in abbrv:
            word = word.upper()
        else:
            word = word.capitalize()
        out.append(word)
    return delim.join(out)


def string_variations(
    names,
    preprocess=["lower", "strip"],
    swaps=[(" ", "_"), (" ", "_"), ("-", " "), ("_", " "), ("-", ""), ("_", "")],
):
    """
    Returns equilvaent string variations based on an input set of strings.

    Parameters
    ----------
    names: {list, str}
        String or list of strings to generate name variations of.
    preprocess: list
        List of preprocessing string functions to apply before generating
        variations.
    swaps: list
        List of tuples for str.replace(out, in).

    Returns
    --------
    set
        Set (or SortedSet, if sortedcontainers installed) of unique string
        variations.
    """
    vars = set()
    # convert input to list if singular
    if isinstance(names, str):
        names = [names]

    swapout = [s[0] for s in swaps]
    for n in names:
        n = str(n)
        for p in preprocess:
            n = getattr(n, p)()
        vars.add(n)
        if any([s in n for s in swapout]):
            vars = vars.union([n.replace(*s) for s in swaps])
    return vars


def parse_entry(
    entry,
    regex=r"(\s)*?(?P<value>[\.\w]+)(\s)*?",
    delimiter=",",
    values_only=True,
    first_only=True,
    errors=None,
    replace_nan="None",
):
    """
    Parses an arbitrary string data entry to return
    values based on a regular expression containing
    named fields including 'value' (and any others).
    If the entry is of non-string type, this will
    return the value (e.g. int, float, NaN, None).

    Parameters
    -----------------------
    entry: str
        String entry which to search for the regex pattern.
    regex: str
        Regular expression to compile and use to search the
        entry for a value.
    delimiter: str, ','
        Optional delimiter to split the string in case of multiple
        inclusion.
    values_only: bool, True
        Option to return only values (single or list), or to instead
        return the dictionary corresponding to the matches.
    first_only: bool, True
        Option to return only the first match, or else all matches.

    Not yet implemented:

    errors: int|float|np.nan|None, None
        Error value to denote 'no match'.
    """

    if isinstance(entry, str):
        pattern = re.compile(regex)
        matches = []
        if not delimiter or (delimiter is None):
            subparts = [entry]
        else:
            subparts = entry.split(delimiter)

        for _l in subparts:
            _m = pattern.match(_l)
            if _m:
                _d = dict(value=_m.group("value"))
                # Add other groups
                _d.update(
                    {
                        k: _m.group(k)
                        for (k, ind) in pattern.groupindex.items()
                        if not k == "value"
                    }
                )

            else:
                _d = dict(value=replace_nan)
                # Add other groups
                _d.update(
                    {
                        k: replace_nan
                        for (k, ind) in pattern.groupindex.items()
                        if not k == "value"
                    }
                )
            matches.append(_d)

        if values_only:
            matches = [m["value"] for m in matches]

        if first_only:
            return matches[0]

        return matches
    else:
        if entry is None:
            entry = replace_nan
        elif isinstance(entry, float):
            if np.isnan(entry):
                entry = replace_nan
        if first_only:
            return entry
        else:
            return [entry]


def split_records(data, delimiter="\r\n"):
    """
    Splits records in a csv where quotation marks are used.
    Splits on a delimiter followed by an even number of quotation marks.
    """
    # https://stackoverflow.com/a/2787979
    return re.split(delimiter + """(?=(?:[^'"]|'[^']*'|"[^"]*")*$)""", data)
