import re
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()

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
