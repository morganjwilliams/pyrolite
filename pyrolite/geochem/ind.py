"""
Collections and indexes of elements and oxides, and functions
for obtaining these and relevant properties (e.g. radii).

Todo
------
* Incompatibility indexes for spider plot ordering.
"""
import re
import numpy as np
import pandas as pd
import periodictable as pt
from tinydb import TinyDB, Query
from ..util.text import titlecase, remove_suffix
from ..util.meta import (
    pyrolite_datafolder,
    sphinx_doi_link,
    update_docstring_references,
)
from ..util.log import Handle

logger = Handle(__name__)

__radii__ = {}


def _load_radii():
    """Import radii tables to a module-level dictionary indexed by reference."""
    global __radii__
    for name in ["shannon", "whittaker_muntus"]:
        pth = (pyrolite_datafolder(subfolder="radii") / "{}.csv".format(name)).resolve()
        assert pth.exists() and pth.is_file()
        df = pd.read_csv(pth).set_index("index", drop=True)
        assert hasattr(df, "element")
        __radii__[name] = df


_load_radii()
########################################################################################


def common_elements(cutoff=92, output="string", order=None, as_set=False):
    """
    Provides a list of elements up to a particular cutoff (by default including U).

    Parameters
    -----------
    cutoff : :class:`int`
        Upper cutoff on atomic number for the output list. Defaults to stopping at
        uranium (92).
    output : :class:`str`
        Whether to return output list as formulae ('formula') or strings (anthing else).
    order : :class:`callable`
        Sorting function for elements.
    as_set : :class:`bool`, :code:`False`
        Whether to return a :class:`set` (:code:`True`) or :class:`list` (:code:`False`).


    Returns
    -------
    :class:`list` | :class:`set`
        List of elements.

    Notes
    ------

    Formulae cannot be used as members of a set, and hence sets returned will
    instead consist only of strings.

    Todo
    -----

    * Implement ordering for e.g. incompatibility.
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


def REE(output="string", dropPm=True):
    """
    Provides a list of Rare Earth Elements.

    Parameters
    -----------
    output : :class:`str`
        Whether to return output list as formulae ('formula') or strings (anthing else).
    dropPm : :class:`bool`
        Whether to exclude the (almost) non-existent element Promethium from the REE
        list.

    Returns
    -------
    :class:`list` | :class:`set`
        List of REE.
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
    if dropPm:
        elements = [i for i in elements if not i == "Pm"]
    if output == "formula":
        elements = [getattr(pt, el) for el in elements]
    return elements


def REY(output="string", dropPm=True):
    """
    Provides a list of Rare Earth Elements, with the addition of Yttrium.

    Parameters
    -----------
    output : :class:`str`
        Whether to return output list as formulae ('formula') or strings (anthing else).

    Returns
    -------
    :class:`list` | :class:`set`
        List of REE+Y.

    Notes
    ------
    This currently modifies the hardcoded list of :func:`REE`, but could be adapated
    for different element ordering.
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
        "Y",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]
    if dropPm:
        elements = [i for i in elements if not i == "Pm"]
    if output == "formula":
        elements = [getattr(pt, el) for el in elements]
    return elements


# this uses unhashable objects in the call, cannot be optimised using LRU cache
def common_oxides(
    elements: list = [],
    output="string",
    addition: list = ["FeOT", "Fe2O3T", "LOI"],
    exclude=["O", "He", "Ne", "Ar", "Kr", "Xe"],
    as_set=False,
):
    """
    Creates a list of oxides based on a list of elements.

    Parameters
    -----------
    elements : :class:`list`, []
        List of elements to obtain oxide forms for.
    output : :class:`str`
        Whether to return output list as formulae ('formula') or strings (anthing else).
    addition : :class:`list`, []
        Additional components to append to the list.
    exclude : :class:`list`
        Elements to not produce oxide forms for (e.g. oxygen, noble gases).
    as_set : :class:`bool`
        Whether to return a :class:`set` (:code:`True`) or :class:`list` (:code:`False`).

    Returns
    -------
    :class:`list` | :class:`set`
        List of oxides.

    Notes
    ------
    Formulae cannot be used as members of a set, and hence sets returned will
    instead consist only of strings.

    Todo
    ----
    * Element verification
    * Conditional additional components on the presence of others (e.g. Fe - FeOT)
    """
    if not elements:
        elements = __common_elements__ - set(exclude)
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
    Creates a list of oxides for a cationic element (oxide of ions with c=1+ and above).

    Parameters
    -----------
    cation : :class:`str` | :class:`periodictable.core.Element`
        Cation to obtain oxide forms for.
    output : :class:`str`
        Whether to return output list as formulae ('formula') or strings (anthing else).

    Returns
    -------
    :class:`list` | :class:`set`
        List of oxides.
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

    if not output == "formula":
        oxides = [str(ox) for ox in oxides]
    return oxides


def get_cations(component: str, exclude=[], total_suffix="T"):
    """
    Returns the principal cations in an oxide component.

    Parameters
    -----------
    component : :class:`str` | :class:`periodictable.formulas.Formula`
        Component to obtain cations for.
    exclude : :class:`list`
        Components to exclude, i.e. anions (e.g. O, Cl, F).

    Returns
    -------
    :class:`list`
        List of cations.

    Todo
    -----
        * Consider implementing :class:`periodictable.core.Element` return.
    """
    if isinstance(component, str):
        component = remove_suffix(component, suffix=total_suffix)

    exclude += ["O"]
    atms = pt.formula(component).atoms
    cations = [el for el in atms.keys() if not el.__str__() in exclude]
    return cations


def get_isotopes(ratio_text):
    """
    Regex for isotope ratios.

    Parameters
    -----------
    ratio_text : :class:`str`
        Text to extract isotope ratio components from.

    Returns
    -----------
    :class:`list`
        Isotope ration numerator and denominator.
    """
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


def by_incompatibility(els, reverse=False):
    """
    Order a list of elements by their relative 'incompatibility' given by
    a proxy of the relative abundances in Bulk Continental Crust over
    a Primitive Mantle Composition.

    Parameters
    ------------
    els : :class:`list`
        List of element names to be reodered.
    reverse : :class:`bool`
        Whether to reverse the ordering.

    Returns
    ---------
    :class:`list`
        Reordered list of elements.

    Notes
    -----
    Some elements are missing from this list, as as such will be omitted.
    """
    incomp = [
        ["Tl", "Cs", "I", "W", "Rb", "Ba", "Th", "Bi", "Pb", "K", "U", "B"],
        ["Sb", "As", "Be", "La", "Ce", "F", "Pr", "Mo", "Ta", "Nd", "Sr", "Nb"],
        ["Zr", "Hf", "Sn", "Br", "Li", "Ag", "Sm", "Na", "Cl", "Eu", "Gd"],
        ["Tb", "P", "Hg", "Dy", "Ho", "Y", "Er", "Lu", "Yb", "Tm", "Ga", "Al"],
        ["Ti", "In", "Cd", "S", "Ca", "Se", "V", "Cu", "Zn", "Sc", "Si", "Ge"],
        ["Fe", "Au", "Mn", "Re", "Co", "Pd", "Pt", "Mg", "Ru", "Cr", "Ni"],
        ["Ir", "Os"],
    ]
    incomp = [i for part in incomp for i in part]
    missing = np.array([el not in incomp for el in els])
    if missing.any():
        logger.warning(
            "Some elements could not be ordered: {}.".format(
                ",".join([els[m] for m in np.argwhere(missing).flatten()])
            )
        )
    if reverse:
        return [i for i in incomp[::-1] if i in els]
    else:
        return [i for i in incomp if i in els]


def by_number(els, reverse=False):
    """
    Order a list of elements by their atomic number.

    Parameters
    ------------
    els : :class:`list`
        List of element names to be reodered.
    reverse : :class:`bool`
        Whether to reverse the ordering.

    Returns
    ---------
    :class:`list`
        Reordered list of elements.
    """
    ordered = np.array(els)[np.argsort([getattr(pt, el).number for el in els])]
    if reverse:
        ordered = ordered[::-1]
    return list(ordered)


# RADII ################################################################################
@update_docstring_references
def get_ionic_radii(
    element,
    charge=None,
    coordination=None,
    variant=[],
    source="shannon",
    pauling=True,
    **kwargs
):
    """
    Function to obtain ionic radii for a given ion and coordination [#ref_1]_
    [#ref_2]_.

    Parameters
    -----------
    element : :class:`str` | :class:`list`
        Element to obtain a radii for. If a list is passed, the function will be applied
        over each of the items.
    charge : :class:`int`
        Charge of the ion to obtain a radii for. If unspecified will use the default
        charge from :mod:`pyrolite.mineral.ions`.
    coordination : :class:`int`
        Coordination of the ion to obtain a radii for.
    variant : :class:`list`
        List of strings specifying particular variants (here 'squareplanar' or
        'pyramidal', 'highspin' or 'lowspin').
    source : :class:`str`
        Name of the data source for ionic radii ('shannon' [#ref_1]_ or
        'whittaker' [#ref_2]_).
    pauling : :class:`bool`
        Whether to use the radii consistent with Pauling (1960) [#ref_3]_ from the
        Shannon (1976) radii dataset [#ref_1]_.

    Returns
    --------
    :class:`pandas.Series` | :class:`float`
        Series with viable ion charge and coordination, with associated radii in
        angstroms. If the ion charge and coordiation are completely specified and
        found in the table, a single value will be returned instead.

    Notes
    ------
    Shannon published two sets of radii. The first ('Crystal Radii') were using
    Shannon's value for :math:`r(O^{2-}_{VI})` of 1.26 Å, while the second
    ('Ionic Radii') is consistent with the Pauling (1960) value of
    :math:`r(O^{2-}_{VI})` of 1.40 Å [#ref_3]_.

    References
    ----------
    .. [#ref_1] Shannon RD (1976). Revised effective ionic radii and systematic
            studies of interatomic distances in halides and chalcogenides.
            Acta Crystallographica Section A 32:751–767.
            doi: shannon1976
    .. [#ref_2] Whittaker, E.J.W., Muntus, R., 1970.
           Ionic radii for use in geochemistry.
           Geochimica et Cosmochimica Acta 34, 945–956.
           doi: whittaker_muntus1970
    .. [#ref_3] Pauling, L., 1960. The Nature of the Chemical Bond.
            Cornell University Press, Ithaca, NY.

    Todo
    -----
    * Implement interpolation for coordination +/- charge.
    """
    if isinstance(element, list):
        return [
            get_ionic_radii(
                e,
                charge=charge,
                coordination=coordination,
                variant=variant,
                source=source,
                pauling=pauling,
                **kwargs
            )
            for e in element
        ]

    if "shannon" in source.lower():
        df = __radii__["shannon"]
        target = ["crystalradius", "ionicradius"][pauling]
    elif "whittaker" in source.lower():
        df = __radii__["whittaker_muntus"]
        target = "ionicradius"
    else:
        raise AssertionError(
            "Invalid `source` argument. Options: {}".format(
                " ,".join("'{}'".format(src) for src in __radii__.keys())
            )
        )

    elfltr = df.element == element
    fltrs = elfltr.copy().astype(int)
    if charge is not None:
        if charge in df.loc[elfltr, "charge"].unique():
            fltrs *= df.charge == charge
        else:
            logging.warn("Charge {:d} not in table.".format(int(charge)))
            # try to interpolate over charge?..
            # interpolate_charge=True
    else:
        charge = getattr(pt, element).default_charge
        fltrs *= df.charge == charge

    if coordination is not None:
        if coordination in df.loc[elfltr, "coordination"].unique():
            fltrs *= df.coordination == coordination
        else:
            logging.warn("Coordination {:d} not in table.".format(int(coordination)))
            # try to interpolate over coordination
            # interpolate_coordination=True

    # assert not interpolate_coordination and interpolate_charge

    if variant:  # todo warning for missing variants
        for v in variant:
            fltrs *= table.variant.apply(lambda x: v in x)

    result = df.loc[fltrs.astype(bool), target]

    if result.index.size == 1:
        return result.values[0]  # return the specific number
    else:
        return result  # return the series


ordering = {"incompatibility": by_incompatibility, "number": by_number}

# update doi links for radii

get_ionic_radii.__doc__ = get_ionic_radii.__doc__.replace(
    "shannon1976", sphinx_doi_link("10.1107/S0567739476001551")
)
get_ionic_radii.__doc__ = get_ionic_radii.__doc__.replace(
    "whittaker_muntus1970", sphinx_doi_link("10.1016/0016-7037(70)90077-3")
)
# generate sets
__db__ = TinyDB(str(pyrolite_datafolder(subfolder="geochem") / "geochemdb.json"))
__common_elements__ = set(__db__.search(Query().name == "elements")[0]["collection"])
__common_oxides__ = set(__db__.search(Query().name == "oxides")[0]["collection"])
__db__.close()
