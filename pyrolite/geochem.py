import pandas as pd
import numpy as np
import mpmath
import periodictable as pt
import matplotlib.pyplot as plt
import functools
import re
from .comp.renorm import renormalise
from .norm import ReferenceCompositions, RefComp, scale_multiplier
from .util.text import titlecase
from .util.pd import to_frame
from .util.math import *
from .util.general import iscollection
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def get_radii(el):
    """Convenience function for ionic radii."""
    if isinstance(el, list):
        return [get_radii(e) for e in el]
    elif not isinstance(el, str):
        el = str(el)
    return _RADII[el]


def ischem(s):
    """
    Checks if a string corresponds to chemical component.
    Here simply checking whether it is a common element or oxide.

    TODO: Implement checking for other compounds, e.g. carbonates.
    """
    chems = common_oxides() + common_elements()
    chems = [e.upper() for e in chems]
    if isinstance(s, list):
        return [str(st).upper() in chems for st in s]
    else:
        return str(s).upper() in chems


def is_isotoperatio(text):
    """Check if text is plausibly an isotope ratio."""
    isotopes = get_isotopes(text)
    return len(isotopes) == 2


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


def repr_isotope_ratio(isotope_ratio):
    """
    Format an isotope ratio pair as a string.

    Parameters
    -----------
    isotope_ratio : tuple
        Numerator, denominator pair.
    """
    num, den = isotope_ratio
    isomatch = r"([0-9][0-9]?[0-9]?)"
    elmatch = r"([a-zA-Z][a-zA-Z]?)"
    num_iso, num_el = re.findall(isomatch, num)[0], re.findall(elmatch, num)[0]
    den_iso, den_el = re.findall(isomatch, den)[0], re.findall(elmatch, den)[0]
    return "{}{}{}{}".format(num_iso, titlecase(num_el), den_iso, titlecase(den_el))


def tochem(strings: list, abbrv=["ID", "IGSN"], split_on="[\s_]+"):
    """
    Converts a list of strings containing come chemical compounds to
    appropriate case.
    """
    # accomodate single string passed
    if not type(strings) in [list, pd.core.indexes.base.Index]:
        strings = [strings]

    # translate elements and oxides
    chems = common_oxides() + common_elements()
    trans = {str(e).upper(): str(e) for e in chems}
    strings = [trans[str(h).upper()] if str(h).upper() in trans else h for h in strings]

    # translate potential isotope ratios
    strings = [
        repr_isotope_ratio(get_isotopes(h)) if is_isotoperatio(h) else h
        for h in strings
    ]
    return strings


def to_molecular(df: pd.DataFrame, renorm=True):
    """
    Converts mass quantities to molar quantities of the same order.
    E.g.:
    mass% --> mol%
    mass-ppm --> mol-ppm
    """
    df = to_frame(df)
    MWs = [pt.formula(c).mass for c in df.columns]
    if renorm:
        return renormalise(df.div(MWs))
    else:
        return df.div(MWs)


def to_weight(df: pd.DataFrame, renorm=True):
    """
    Converts molar quantities to mass quantities of the same order.
    E.g.:
    mol% --> mass%
    mol-ppm --> mass-ppm
    """
    df = to_frame(df)
    MWs = [pt.formula(c).mass for c in df.columns]
    if renorm:
        return renormalise(df.multiply(MWs))
    else:
        return df.multiply(MWs)


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


def common_elements(cutoff=92, output="string", order=None):
    """
    Provides a list of elements up to a particular cutoff (default: including U)
    Output options are 'formula', or 'string'.

    Todo: implement ordering for e.g. incompatibility.
    """
    elements = [el for el in pt.elements if not (str(el) == "n" or el.number > cutoff)]

    if order is not None:
        sort_function = order
        elements = elements.sort(key=sort_function)

    if not output == "formula":
        elements = [str(el) for el in elements]

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
    if output != "formula":
        oxides = [str(ox) for ox in oxides] + addition
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
    if not output == "formula":
        oxides = [str(ox) for ox in oxides]
    return oxides


def devolatilise(
    df: pd.DataFrame,
    exclude=["H2O", "H2O_PLUS", "H2O_MINUS", "CO2", "LOI"],
    renorm=True,
):
    """
    Recalculates components after exclusion of volatile phases (e.g. H2O, CO2).
    """
    keep = [i for i in df.columns if not i in exclude]
    if renorm:
        return renormalise(df.loc[:, keep])
    else:
        return df.loc[:, keep]


def oxide_conversion(oxin, oxout):
    """
    Generates a function to convert oxide components between
    two elemental oxides, for use in redox recalculations.
    """
    if not (isinstance(oxin, pt.formulas.Formula) or isinstance(oxin, pt.core.Element)):
        oxin = pt.formula(oxin)
    if not (
        isinstance(oxout, pt.formulas.Formula) or isinstance(oxout, pt.core.Element)
    ):
        oxout = pt.formula(oxout)

    inatoms = {k: v for (k, v) in oxin.atoms.items() if not k.__str__() == "O"}
    in_els = inatoms.keys()
    outatoms = {k: v for (k, v) in oxout.atoms.items() if not k.__str__() == "O"}
    out_els = outatoms.keys()
    assert len(inatoms) == len(outatoms) == 1  # Assertion of simple oxide
    assert in_els == out_els  # Need to be dealilng with the same element!
    # Moles of product vs. moles of reactant
    cation_coefficient = list(inatoms.values())[0] / list(outatoms.values())[0]

    def convert_series(dfser: pd.Series, molecular=False):
        if molecular:
            factor = cation_coefficient
        else:
            factor = cation_coefficient * oxout.mass / oxin.mass
        converted = dfser * factor
        return converted

    doc = "Convert series from " + str(oxin) + " to " + str(oxout)
    convert_series.__doc__ = doc
    return convert_series


def recalculate_redox(
    df: pd.DataFrame, to_oxidised=False, renorm=True, total_suffix="T", logdata=False
):
    """
    Recalculates abundances of redox-sensitive components (particularly Fe),
    and normalises a dataframe to contain only one oxide species for a given
    element.

    Consider reimplementing total suffix as a lambda formatting function
    to deal with cases of prefixes, capitalisation etc.

    Automatic generation of multiple redox species from dataframes
    would also be a natural improvement.

    # todo: update to incorporate Fe and transformation from multiple oxides to one
    """
    # Assuming either (a single column) or (FeO + Fe2O3) are reported
    # Fe columns - FeO, Fe2O3, FeOT, Fe2O3T
    FeO = pt.formula("FeO")
    Fe2O3 = pt.formula("Fe2O3")
    dfc = df.copy(deep=True)
    ox_species = ["Fe2O3", "Fe2O3" + total_suffix]
    ox_in_df = [i for i in ox_species if i in dfc.columns]
    red_species = ["FeO", "FeO" + total_suffix]
    red_in_df = [i for i in red_species if i in dfc.columns]
    if logdata:
        dfc.loc[:, ox_in_df + red_in_df] = dfc.loc[:, ox_in_df + red_in_df].applymap(
            np.exp
        )
    if to_oxidised:
        key = "Fe2O3T"
        oxFe = oxide_conversion(FeO, Fe2O3)
        Fe2O3T = dfc.loc[:, ox_in_df].fillna(0).sum(axis=1) + oxFe(
            dfc.loc[:, red_in_df].fillna(0)
        ).sum(axis=1)
        dfc.loc[:, key] = Fe2O3T
        Fe2O3T[Fe2O3T <= 0] = np.nan
        to_drop = red_in_df + [i for i in ox_in_df if not i.endswith(total_suffix)]
    else:
        key = "FeOT"
        reduceFe = oxide_conversion(Fe2O3, FeO)
        FeOT = dfc.loc[:, red_in_df].fillna(0).sum(axis=1) + reduceFe(
            dfc.loc[:, ox_in_df].fillna(0)
        ).sum(axis=1)
        FeOT[FeOT <= 0] = np.nan
        dfc.loc[:, key] = FeOT
        to_drop = ox_in_df + [i for i in red_in_df if not i.endswith(total_suffix)]

    if logdata:
        dfc.loc[:, key] = np.exp(dfc.loc[:, key].values)

    dfc = dfc.drop(columns=to_drop)

    if renorm:
        return renormalise(dfc)
    else:
        return dfc


def aggregate_cation(
    df: pd.DataFrame,
    cation=None,
    oxide=None,
    form="oxide",
    unit_scale=scale_multiplier("Wt%", "Wt%"),
    logdata=False,
):
    """
    Aggregates cation information from oxide and elemental components
    to a single series. Allows simultaneous scaling (e.g. from ppm to wt%).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for which to aggregate cation data.
    cation : str
        Name of cation to aggregate.
    oxide:
        Name of oxide to aggregate.
    form: {'oxide', 'element'}
        Whether to aggregate to oxide or elemental form.
    unit_scale:
        The scale factor difference between the components. Unity if both have the same
        units. Can be converted using scale_multiplier: e.g.
        scale_multiplier("Wt%", "ppm")


    Todo
    -------
        Needs to also implement a 'molecular' version.

    """
    dfc = df.copy()
    # Should first check that neither the element or oxide is present more than once
    assert not ((cation is None) and (oxide is None))
    if cation is not None and oxide is not None:
        elstr = str(cation)
        oxstr = str(oxide)
    elif oxide is not None:
        oxstr = str(oxide)
        elstr = str(get_cations(oxide)[0])
    elif cation is not None:
        elstr = str(cation)
        oxstr = [o for o in dfc.columns if o in simple_oxides(elstr)][0]
        assert oxstr, "Oxidation state uknown. Please specify desired oxide."

    el, ox = pt.formula(elstr), pt.formula(oxstr)

    for c in [elstr, oxstr]:
        if not c in df.columns:
            logger.info("Adding {} column.".format(c))
            dfc[c] = np.nan

    eldata = dfc.loc[:, elstr].values
    oxdata = dfc.loc[:, oxstr].values
    if logdata:
        eldata = np.exp(eldata)
        oxdata = np.exp(oxdata)

    if form == "oxide":
        if unit_scale is None:
            unit_scale = 1.0
        assert unit_scale > 0
        convert_function = oxide_conversion(ox, el)
        conv_values = convert_function(eldata) * unit_scale
        totals = np.nansum(np.vstack((oxdata, conv_values)), axis=0)
    elif form == "element":
        if unit_scale is None:
            unit_scale = 1.0
        assert unit_scale > 0
        convert_function = oxide_conversion(el, ox)
        conv_values = convert_function(oxdata) * unit_scale
        totals = np.nansum(np.vstack((eldata, conv_values)), axis=0)

    totals[np.isclose(totals, 0)] = np.nan

    if logdata:
        totals = np.log(totals)

    if form == "oxide":
        dfc.loc[:, oxstr] = totals
        dfc.drop(columns=[elstr], inplace=True)
        assert elstr not in dfc.columns
    else:
        dfc.loc[:, elstr] = totals
        dfc.drop(columns=[oxstr], inplace=True)
        assert oxstr not in dfc.columns

    return dfc


def check_multiple_cation_inclusion(df, exclude=["LOI", "FeOT", "Fe2O3T"]):
    """
    Returns cations which are present in both oxide and elemental form.

    Todo: Options for output (string/formula).
    """
    major_components = [i for i in common_oxides() if i in df.columns]
    elements_as_majors = [
        get_cations(oxide)[0] for oxide in major_components if not oxide in exclude
    ]
    elements_as_traces = [
        c for c in common_elements(output="formula") if str(c) in df.columns
    ]
    return set([el for el in elements_as_majors if el in elements_as_traces])


def convert_chemistry(df, columns=[], logdata=False, renorm=False):
    """
    Tries to convert a dataframe with one set of components to another.

    Parameters
    -----------
    df : pd.DataFrame
        Dataframe to convert.
    columns : list, set
        Set of columns to try to extract from the dataframe.
    """
    df = df.copy()
    current = df.columns
    ok = [i for i in columns if i in current]
    get = [i for i in columns if i not in current]
    multiples = check_multiple_cation_inclusion(df)
    oxides = common_oxides(addition=[])
    elements = common_elements()
    Fe_parts = ["Fe", "FeO", "Fe2O3", "Fe2O3T", "FeOT"]

    # Aggregate the columns which are otherwise OK
    for o in ok:
        if o in oxides + elements:
            elem = get_cations(o)[0]
            if elem in multiples:
                if o in oxides:
                    df = aggregate_cation(
                        df, cation=elem, oxide=o, form="oxide", logdata=logdata
                    )
                    logger.info("Transforming {} to {}".format(elem, o))
                else:
                    potential_oxides = simple_oxides(o)
                    present_oxides = [p for p in potential_oxides if p in current]
                    for ox in present_oxides:  # aggregate all the relevant oxides
                        df = aggregate_cation(
                            df, cation=o, oxide=ox, form="element", logdata=logdata
                        )
                        logger.info("Transforming {} to {}".format(ox, o))
        if o in Fe_parts:
            pass

    # --- Try to get the new columns ----
    for g in get:
        if g in oxides:
            elem = get_cations(g)[0]
            oxide = g
            logger.info("Getting {oxide} from {elem}".format(oxide=oxide, elem=elem))
            df = aggregate_cation(
                df, cation=elem, oxide=oxide, form="oxide", logdata=logdata
            )

        elif g in elements:
            elem = g
            potential_oxides = simple_oxides(g)
            present_oxides = [p for p in potential_oxides if p in current]
            for ox in present_oxides:  # aggregate all the relevant oxides
                logger.info("Getting {elem} from {oxide}".format(oxide=ox, elem=elem))
                df = aggregate_cation(
                    df, cation=elem, oxide=ox, form="element", logdata=logdata
                )

    # --- Try to get the new columns - iron redox section ----
    for g in get:
        if g in Fe_parts:
            current_Fe = [i for i in Fe_parts if i in df.columns]
            c_fe_str = ", ".join(current_Fe)
            if g in ["FeO", "FeOT"]:
                logger.info("Reducing {} to {}.".format(c_fe_str, g))
                df = recalculate_redox(
                    df, to_oxidised=False, renorm=False, logdata=logdata
                )
            elif g in ["Fe2O3", "Fe2O3T"]:
                logger.info("Oxidising {} to {}.".format(c_fe_str, g))
                df = recalculate_redox(
                    df, to_oxidised=True, renorm=False, logdata=logdata
                )
            elif g in ["Fe"]:
                to_oxidised = False
                logger.info("Oxidising {} to {}.".format(c_fe_str, g))
                df = recalculate_redox(
                    df, to_oxidised=False, renorm=False, logdata=logdata
                )
                df = aggregate_cation(df, cation=g, form="element")

    if renorm:
        return renormalise(df.loc[:, columns])
    else:
        return df.loc[:, columns]


def add_ratio(
    df: pd.DataFrame, ratio: str, alias: str = "", norm_to=None, convert=lambda x: x
):
    """
    Add a ratio of components A and B, given in the form of string 'A/B'.
    Returned series be assigned an alias name.

    Parameters
    -----------
    df: pd.DataFrame
        Dataframe to append ratio to.
    ratio: str
        String decription of ratio in the form A/B[_n].
    alias: str
        Alternate name for ratio to be used as column name.
    norm_to: {None, RefComp, str}
        Reference composition to normalise to.
    convert:
        Data processing function to be calculated prior to ratio.
    """

    num, den = ratio.split("/")
    _to_norm = False
    if den.lower().endswith("_n"):
        den = titlecase(den.lower().replace("_n", ""))
        _to_norm = True
    assert titlecase(num) in df.columns
    assert titlecase(den) in df.columns

    if _to_norm or (norm_to is not None):
        if isinstance(norm_to, str):
            norm = ReferenceCompositions()[norm_to]
            num_n, den_n = norm[num].value, norm[den].value
        elif isinstance(norm_to, RefComp):
            num_n, den_n = norm_to[num].value, norm_to[den].value
        elif iscollection(norm_to):  # list, iterable, pd.Index etc
            num_n, den_n = norm_to
        else:
            norm = ReferenceCompositions()["Chondrite_PON"]
            num_n, den_n = norm[num].value, norm[den].value

    name = [ratio if not alias else alias][0]
    conv = convert(df.loc[:, [num, den]])
    conv.loc[(conv[den] == 0.0) | (conv[num] == 0.0), den] = np.nan  # avoid 0, inf
    df.loc[:, name] = conv.loc[:, num] / conv.loc[:, den]
    return df


def add_MgNo(df: pd.DataFrame, molecularIn=False, elemental=False, components=False):

    if not molecularIn:
        if components:
            # Iron is split into species
            df.loc[:, "Mg#"] = (
                df["MgO"]
                / pt.formula("MgO").mass
                / (
                    df["MgO"] / pt.formula("MgO").mass
                    + df["FeO"] / pt.formula("FeO").mass
                )
            )
        else:
            # Total iron is used
            assert "FeOT" in df.columns
            df.loc[:, "Mg#"] = (
                df["MgO"]
                / pt.formula("MgO").mass
                / (
                    df["MgO"] / pt.formula("MgO").mass
                    + df["FeOT"] / pt.formula("FeO").mass
                )
            )
    else:
        if not elemental:
            # Molecular Oxides
            df.loc[:, "Mg#"] = df["MgO"] / (df["MgO"] + df["FeO"])
        else:
            # Molecular Elemental
            df.loc[:, "Mg#"] = df["Mg"] / (df["Mg"] + df["Fe"])


def lambda_lnREE(
    df,
    norm_to="Chondrite_PON",
    exclude=["Pm", "Eu"],
    params=None,
    degree=5,
    append=[],
    **kwargs
):
    """
    Calculates lambda coefficients for a given set of REE data, normalised
    to a specific composition. Lambda factors are given for the
    radii vs. ln(REE/NORM) polynomical combination.

    TODO: Operate only on valid rows.
    """
    non_null_cols = df.columns[~df.isnull().all(axis=0)]
    ree = [
        i
        for i in REE()
        if i in df.columns
        and (not str(i) in exclude)
        and (str(i) in non_null_cols or i in non_null_cols)
    ]  # no promethium
    radii = np.array(get_radii(ree))

    if params is None:
        params = OP_constants(radii, degree=degree)
    else:
        degree = len(params)

    null_in_row = pd.isnull(df.loc[:, ree]).any(axis=1)
    norm_df = df.loc[~null_in_row, ree].copy()  # initialize normdf

    labels = [chr(955) + str(d) for d in range(degree)]

    if norm_to is not None:  # None = already normalised data
        if isinstance(norm_to, str):
            norm = ReferenceCompositions()[norm_to]
            norm_abund = np.array([norm[str(el)].value for el in ree])
        elif isinstance(norm_to, RefComp):
            norm_abund = np.array([getattr(norm_to, str(e)) for e in ree])
        else:  # list, iterable, pd.Index etc
            norm_abund = np.array([i for i in norm_abund])
            assert len(norm_abund) == len(ree)

        norm_df.loc[:, ree] = np.divide(norm_df.loc[:, ree].values, norm_abund)

    norm_df.loc[(norm_df <= 0.0).any(axis=1), :] = np.nan  # remove zero or below
    norm_df.loc[:, ree] = norm_df.loc[:, ree].applymap(np.log)

    lambdadf = pd.DataFrame(index=df.index, columns=labels)
    lambda_partial = functools.partial(
        lambdas, xs=radii, params=params, degree=degree, **kwargs
    )  # pass kwargs to lambdas
    # apply along rows
    lambdadf.loc[~null_in_row, labels] = np.apply_along_axis(
        lambda_partial, 1, norm_df.values
    )
    lambdadf.loc[(lambdadf == 0.0).all(axis=1), :] = np.nan
    if append:
        # append the smooth f(radii) function to the dataframe
        func_partial = functools.partial(
            lambda_poly_func, pxs=radii, params=params, degree=degree
        )
        if "function" in append:
            lambdadf["lambda_poly_func"] = np.apply_along_axis(
                func_partial, 1, lambdadf.values
            )

    lambdadf = lambdadf.apply(pd.to_numeric, errors="coerce")
    assert lambdadf.index.size == df.index.size
    return lambdadf


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
