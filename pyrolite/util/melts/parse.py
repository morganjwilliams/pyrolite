import re
import logging
import periodictable as pt
from pyrolite.mineral.mineral import merge_formulae

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def from_melts_cstr(composition_str, formula=True):
    """
    Parses melts composition strings to dictionaries or formulae.

    Parameters
    -----------
    composition_str : :class:`str`
        Composition to parse.
    formula : :class:`bool`
        Whether to output a :class:`periodictable.formula.Formula`


    Todo
    ------
        * Enable parsing of nested brackets in composition.
    """
    regex = r"""(?P<el>[a-zA-Z'[^.-]]+)(?P<num>[^a-zA-Z()]+)"""
    sub = r"""[\']+"""

    def repl(m):
        return "{" + str(m[0].count("""'""")) + "+" + "}"  # replace ' with count(')

    tfm = lambda s: re.sub(sub, repl, s)
    if not formula:
        result = re.findall(regex, composition_str)
        result = [(tfm(el), float(val)) for el, val in result]
        return {k: v for k, v in result}
    else:
        # entries with rounding errors giving negative 0 won't match
        result = tfm(composition_str.replace("-0.0", "0.0"))
        return pt.formula(result)
