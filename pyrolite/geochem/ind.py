import re
import periodictable as pt
from pyrolite.util.text import titlecase

def common_elements(cutoff=92, output="string", order=None, as_set=False):
    """
    Provides a list of elements up to a particular cutoff (default: including U)
    Output options are 'formula', or 'string'.

    Todo: implement ordering for e.g. incompatibility.
    """
    elements = [el for el in pt.elements if not (str(el) == "n" or el.number > cutoff)]

    if as_set:
        return set(map(str, elements))
    else:
        if not output == "formula":
            elements = list(map(str, elements))

        if order is not None:
            sort_function = order
            elements = list(elements).sort(key=sort_function)

        return elements


def REE(output="string", include_extras=False):
    """
    Provides the list of Rare Earth Elements
    Output options are 'formula', or strings.

    Todo: add include extras such as Y.
    """
    elements = [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]
    if output == "formula":
        elements = [getattr(pt, el) for el in elements]
    return elements


def common_oxides(
    elements: list = [],
    output="string",
    addition: list = ["FeOT", "Fe2O3T", "LOI"],
    exclude=["O", "He", "Ne", "Ar", "Kr", "Xe"],
    as_set=False,
):
    """
    Creates a list of oxides based on a list of elements.
    Output options are 'formula', or strings.

    Note: currently return FeOT and LOI even for element lists
    not including iron or water - potential upgrade!

    Todo: element verification
    """
    if not elements:
        elements = [
            el for el in common_elements(output="formula") if not str(el) in exclude
        ]
    else:
        # Check that all elements input are indeed elements..
        pass

    oxides = [ox for el in elements for ox in simple_oxides(el, output=output)]

    if as_set:
        return set(map(str, oxides + addition))
    else:
        if output != "formula":
            oxides = list(map(str, oxides + addition))
        return oxides


def simple_oxides(cation, output="string"):
    """
    Creates a list of oxides for a cationic element
    (oxide of ions with c=1+ and above).
    """
    try:
        if not isinstance(cation, pt.core.Element):
            catstr = titlecase(cation)  # edge case of lowercase str such as 'cs'
            cation = getattr(pt, catstr)
    except AttributeError:
        raise Exception("You must select a cation to obtain oxides.")
    ions = [c for c in cation.ions if c > 0]  # Use only positive charges

    # for 3.6+, could use f'{cation}{1}O{c//2}',  f'{cation}{2}O{c}'
    oxides = [
        str(cation) + str(1) + "O" + str(c // 2)
        if not c % 2
        else str(cation) + str(2) + "O" + str(c)
        for c in ions
    ]
    oxides = [pt.formula(ox) for ox in oxides]
    import periodictable

    if not output == "formula":
        oxides = [str(ox) for ox in oxides]
    return oxides


def get_radii(el):
    """Convenience function for ionic radii."""
    if isinstance(el, list):
        return [get_radii(e) for e in el]
    elif not isinstance(el, str):
        el = str(el)
    return _RADII[el]


def get_cations(oxide: str, exclude=[]):
    """
    Returns the principal cations in an oxide component.

    Todo: Consider implementing periodictable style return.
    """
    if "O" not in exclude:
        exclude += ["O"]
    atms = pt.formula(oxide).atoms
    cations = [el for el in atms.keys() if not el.__str__() in exclude]
    return cations


def get_isotopes(ratio_text):
    """Regex for isotope ratios."""
    forward_isotope = r"([a-zA-Z][a-zA-Z]?[0-9][0-9]?[0-9]?)"
    backward_isotope = r"([0-9][0-9]?[0-9]?[a-zA-Z][a-zA-Z]?)"
    fw = re.findall(forward_isotope, ratio_text)
    bw = re.findall(backward_isotope, ratio_text)
    lfw, lbw = len(fw), len(bw)
    if (lfw > 1 and lbw > 1) or ((lfw < 2) and (lbw < 2)):
        return []
    elif lfw == 2:
        return fw
    elif lbw == 2:
        return bw

__common_oxides__ = common_oxides(as_set=True)
__common_elements__ = common_elements(as_set=True)
__REE = REE()
_RADII = {
    str(k): v
    for (k, v) in zip(
        REE(),
        [
            1.160,
            1.143,
            1.126,
            1.109,
            1.093,
            1.079,
            1.066,
            1.053,
            1.040,
            1.027,
            1.015,
            1.004,
            0.994,
            0.985,
            0.977,
        ],
    )
}
