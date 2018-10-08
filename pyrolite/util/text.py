import re
import textwrap
import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()


def to_width(multiline_string, width=79, **kwargs):
    """Uses builtin textwapr for text wrapping to a specific width."""
    return textwrap.fill(multiline_string, width, **kwargs)


def normalise_whitespace(strg):
    """Substitutes extra tabs, newlines etc. for a single space."""
    return re.sub('\s+', ' ', strg).strip()


def remove_prefix(z, prefix):
    """Remove a specific prefix from the start of a string."""
    if z.startswith(prefix):
        return re.sub("^{}".format(prefix), "", z)
    else:
        return z


def quoted_string(s):
    #if " " in s or '-' in s or '_' in s:
    s = '''"{}"'''.format(s)
    return s


def titlecase(s,
              exceptions=['and', 'in', 'a'],
              abbrv=['ID', 'IGSN', 'CIA', 'CIW',
                     'PIA', 'SAR', 'SiTiIndex', 'WIP'],
              capitalize_first=True,
              split_on='[\s_-]+',
              delim=""):
    """
    Formats strings in CamelCase, with exceptions for simple articles
    and omitted abbreviations which retain their capitalization.
    TODO: Option for retaining original CamelCase.
    """
    # Check if abbrv in string, in which case it'll need to be split first?
    words = re.split(split_on, s)
    out=[]
    first = words[0]
    if capitalize_first and not (first in abbrv):
        first = first.capitalize()

    out.append(first)
    for word in words[1:]:
        if word in exceptions+abbrv:
            pass
        elif word.upper() in abbrv:
            word = word.upper()
        else:
            word = word.capitalize()
        out.append(word)
    return delim.join(out)


def parse_entry(entry,
                regex,
                delimiter=',',
                values_only=True,
                errors=None):
    """
    Parses an arbitrary string data entry to return
    values based on a regular expression containing
    named fields including 'value' (and any others).

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
    values_only: bool, True,
        Option to return only values (single or list), or to instead
        return the dictionary corresponding to the matches.
    errors: int|float|np.nan|None, None
        Error value to denote 'no match'.
    """
    pattern = re.compile(regex)
    matches = []
    for _l in entry.split(delimiter):
        _m = pattern.match(_l)
        if _m:
            _d = dict(value=_m.group('value'))
            # Add other groups
            _d.update({k: _m.group(k)
                       for (k, ind) in pattern.groupindex.items()
                       if not k=='value'})

        else:
            _d = dict(value=None)
            # Add other groups
            _d.update({k: None
                       for (k, ind) in pattern.groupindex.items()
                       if not k=='value'})
        matches.append(_d)
    if values_only:
        matches = [m['value'] for m in matches]
        if len(matches)==1:
            return matches[0]
    return matches
