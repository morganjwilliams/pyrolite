import re
import textwrap
from collections.abc import Callable
from string import ascii_lowercase

import numpy as np

from .log import Handle

logger = Handle(__name__)


def to_width(multiline_string: str, width: int = 79, **kwargs) -> str:
    """Uses builtin textwapr for text wrapping to a specific width."""
    return textwrap.fill(multiline_string, width, **kwargs)


def normalise_whitespace(strg: str) -> str:
    """Substitutes extra tabs, newlines etc. for a single space."""
    return re.sub(r"\s+", " ", strg).strip()


def remove_prefix(z: str, prefix: str) -> str:
    """Remove a specific prefix from the start of a string."""
    if z.startswith(prefix):
        return re.sub(rf"^{prefix}", "", z)
    else:
        return z


def remove_suffix(x: str, suffix: str = " ") -> str:
    """
    Remove a specific suffix from the end of a string.
    """
    x = x.removesuffix(suffix)
    return x


def quoted_string(s: str) -> str:
    # if " " in s or '-' in s or '_' in s:
    s = f'''"{s}"'''
    return s


def titlecase(
    s: str,
    exceptions: list[str] | None = None,
    abbrv: list[str] | None = None,
    capitalize_first: bool = True,
    split_on: str = r"[\.\s_-]+",
    delim: str = "",
) -> str:
    """
    Formats strings in CamelCase, with exceptions for simple articles
    and omitted abbreviations which retain their capitalization.

    Todo
    -----
    * Option for retaining original CamelCase.
    """
    # Check if abbrv in string, in which case it'll need to be split first?
    if abbrv is None:
        abbrv = ["ID", "IGSN", "CIA", "CIW", "PIA", "SAR", "SiTiIndex", "WIP"]
    if exceptions is None:
        exceptions = ["and", "in", "a"]
    words = re.split(split_on, s)
    out = []
    first = words[0]
    if capitalize_first and first not in abbrv:
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
    names: list[str],
    preprocess: list[str] | None = None,
    swaps: list[tuple[str, str]] | None = None,
):
    """
    Returns equilvaent string variations based on an input set of strings.

    Parameters
    ----------
    names: list[str]
        String or list of strings to generate name variations of.
    preprocess: list
        List of preprocessing string methods to apply before generating
        variations.
    swaps: list
        List of tuples for str.replace(out, in).

    Returns
    --------
    set
        Set of unique string variations.
    """
    if swaps is None:
        swaps = [(" ", "_"), (" ", "_"), ("-", " "), ("_", " "), ("-", ""), ("_", "")]
    if preprocess is None:
        preprocess = ["lower", "strip"]
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
        if any(s in n for s in swapout):
            vars = vars.union([n.replace(*s) for s in swaps])
    return vars


def parse_entry(
    entry: str | None | float,
    regex=r"(\s)*?(?P<value>[\.\w]+)(\s)*?",
    delimiter=",",
    values_only=True,
    first_only=True,
    replace_nan="None",
) -> list[dict[str, str]] | dict[str, str] | list[str | float] | str | float:
    """
    Parses an arbitrary string data entry to return
    values based on a regular expression containing
    named fields including 'value' (and any others).
    If the entry is of non-string type, this will
    return the value (e.g. int, float, NaN, None).

    Parameters
    -----------------------
    entry : str
        String entry which to search for the regex pattern.
    regex : str
        Regular expression to compile and use to search the
        entry for a value.
    delimiter : str
        Optional delimiter to split the string in case of multiple
        inclusion.
    values_only : bool
        Option to return only values (single or list), or to instead
        return the dictionary corresponding to the matches.
    first_only : bool
        Option to return only the first match, or else all matches
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
                _d = {"value": _m.group("value")}
                # Add other groups
                _d.update(
                    {
                        k: _m.group(k)
                        for (k, ind) in pattern.groupindex.items()
                        if k != "value"
                    }
                )

            else:
                _d = {"value": replace_nan}
                # Add other groups
                _d.update(
                    {
                        k: replace_nan
                        for (k, ind) in pattern.groupindex.items()
                        if k != "value"
                    }
                )
            matches.append(_d)

        if values_only:
            matches = [m["value"] for m in matches]

        if first_only:
            return matches[0]

        return matches
    else:
        if entry is None or isinstance(entry, float) and np.isnan(entry):
            entry = replace_nan
        if first_only:
            return entry
        else:
            return [entry]


def split_records(data: str, delimiter: str = r"\r\n") -> list[str]:
    """
    Splits records in a csv where quotation marks are used.
    Splits on a delimiter followed by an even number of quotation marks.
    """
    # https://stackoverflow.com/a/2787979
    return re.split(delimiter + """(?=(?:[^'"]|'[^']*'|"[^"]*")*$)""", data)


def slugify(value: str, delim: str = "-") -> str:
    """
    Normalizes a string, removes non-alpha characters, converts spaces to delimiters.

    Parameters
    -----------
    value : str
        String to slugify.
    delim : str
        Delimiter to replace whitespace with.

    Returns
    -------
    str
    """
    return re.sub(r"[-\s]+", delim, re.sub(r"[^\w\s-]", "", value).strip())


def int_to_alpha(num: int) -> str:
    """
    Encode an integer into alpha characters, useful for sequences of axes/figures.

    Parameters
    ----------
    int : int
        Integer to encode.

    Returns
    -------
    str
        Alpha-encoding of a small integer.

    """
    remainder = num
    text = []
    if num >= 26:
        major = remainder // 26
        text.append(ascii_lowercase[remainder // 26 - 1])
        remainder -= major * 26
    text.append(ascii_lowercase[remainder])
    return "".join(text)
