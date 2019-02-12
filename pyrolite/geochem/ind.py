import re
from pathlib import Path
import pandas as pd
import periodictable as pt
from pyrolite.util.text import titlecase
from scipy.interpolate import interp1d
from pyrolite.mineral import ions
from pyrolite.util.general import pyrolite_datafolder
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


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


def get_ionic_radii(element, charge=None, coordination=None, variant=[], pauling=True):
    """
    Function to obtain Shannon's radii for a given ion. Shannon published two sets of
    radii. The first ('Crystal Radii') were using Shannon's value for r($O^{2-}_{VI}$)
    of 1.26 $\AA$, while the second ('Ionic Radii') is consistent with the
    Pauling (1960) value of r($O^{2-}_{VI}$) of 1.40 $\AA$; see `pauling` below.

    Parameters
    -----------
    element : str | list-like
        Element to obtain a radii for. If a list is passed, the function will be applied
        over each of the items.
    charge : int
        Charge of the ion to obtain a radii for. If unspecified will use the default
        charge from :mod:`pyrolite.mineral.ions`.
    coordination : int
        Coordination of the ion to obtain a radii for.
    variant : list
        List of strings specifying particular variants (here 'squareplanar' or
        'pyramidal', 'highspin' or 'lowspin').
    pauling : bool
        Whether to use the radii consistent with Pauling (1960).

    Returns
    --------
    :class:`pandas.Series`
        Dataframe with viable ion charge and coordination, with associated radii in
        angstroms. If the ion charge and coordiation are specified and found in the
        table, a single value will be returned instead.

    Note
    -----
        **Reference**: Shannon RD (1976). Revised effective ionic radii and systematic
        studies of interatomic distances in halides and chalcogenides.
        Acta Crystallographica Section A 32:751–767. doi: 10.1107/S0567739476001551


    Todo
    -----
        * Implement interpolation for coordination +/- charge.
        * Finish Shannon Radii tests
    """
    if isinstance(element, list):
        return [
            get_ionic_radii(
                e,
                charge=charge,
                coordination=coordination,
                variant=variant,
                pauling=pauling,
            )
            for e in element
        ]

    target = ["crystalradius", "ionicradius"][pauling]

    elfltr = __shannon__.element == element
    fltrs = elfltr.copy()
    if charge is not None:
        if charge in __shannon__.loc[elfltr, "charge"].unique():
            fltrs *= __shannon__.charge == charge
        else:
            logging.warn("Charge {:d} not in table.".format(int(charge)))
            # try to interpolate over charge?..
            # interpolate_charge=True
    else:
        charge = getattr(pt, element).default_charge
        fltrs *= __shannon__.charge == charge

    if coordination is not None:
        if coordination in __shannon__.loc[elfltr, "coordination"].unique():
            fltrs *= __shannon__.coordination == coordination
        else:
            logging.warn("Coordination {:d} not in table.".format(int(coordination)))
            # try to interpolate over coordination
            # interpolate_coordination=True

    # assert not interpolate_coordination and interpolate_charge

    if variant:  # todo warning for missing variants
        for v in variant:
            fltrs *= __shannon__.variant.apply(lambda x: v in x)

    result = __shannon__.loc[fltrs, target]
    if result.index.size == 1:
        return result.values[0]  # return the specific number
    else:
        return result  # return the series


__shannon__ = pd.read_csv(pyrolite_datafolder("shannon") / "radii.csv").set_index(
    "index", drop=True
)

# private sets to improve performance
__common_oxides__ = common_oxides(as_set=True)
__common_elements__ = common_elements(as_set=True)
__REE = REE()

_RADII = {
    str(k): v
    for (k, v) in zip(
        REE(), [get_ionic_radii(e, charge=3, coordination=8) for e in REE()]
    )
}
