import pandas as pd
import numpy as np
import mpmath
import periodictable as pt
import matplotlib.pyplot as plt

from .compositions import renormalise
from .normalisation import ReferenceCompositions
from .util.text import titlecase
from .util.pd import to_frame
from .util.math import OP_constants, lambdas, lambda_poly, weighted_comb, \
                       expand_lambdas
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
    chems = common_oxides(output='str') + common_elements(output='str')
    chems = [e.upper() for e in chems]
    if isinstance(s, list):
        return [str(st).upper() in chems for st in s]
    else:
        return str(s).upper() in chems


def tochem(strings:list, abbrv=['ID', 'IGSN'], split_on='[\s_]+'):
    """
    Converts a list of strings containing come chemical compounds to
    appropriate case.
    """
     # accomodate single string passed
    if not type(strings) in [list, pd.core.indexes.base.Index]:
        strings = [strings]
    chems = common_oxides(output='str') + common_elements(output='str')
    trans = {str(e).upper(): str(e) for e in chems}
    strings = [trans[str(h).upper()]
               if str(h).upper() in trans else h
               for h in strings]
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


def get_cations(oxide:str, exclude=[]):
    """
    Returns the principal cations in an oxide component.

    Todo: Consider implementing periodictable style return.
    """
    if 'O' not in exclude:
        exclude += ['O']
    atms = pt.formula(oxide).atoms
    cations = [el for el in atms.keys() if not el.__str__() in exclude]
    return cations


def common_elements(cutoff=92, output='formula'):
    """
    Provides a list of elements up to a particular cutoff (default: including U)
    Output options are 'formula', or strings.
    """
    elements = [el for el in pt.elements
                if not (el.__str__() == 'n' or el.number>cutoff)]
    if not output == 'formula':
        elements = [el.__str__() for el in elements]
    return elements


def REE(output='formula', include_extras=False):
    """
    Provides the list of Rare Earth Elements
    Output options are 'formula', or strings.

    Todo: add include extras such as Y.
    """
    elements = ['La', 'Ce', 'Pr', 'Nd', 'Pm',
                'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    if output == 'formula':
        elements = [getattr(pt, el) for el in elements]
    return elements


def common_oxides(elements: list=[], output='formula',
                  addition: list=['FeOT', 'Fe2O3T', 'LOI'],
                  exclude=['O', 'He', 'Ne', 'Ar', 'Kr', 'Xe']):
    """
    Creates a list of oxides based on a list of elements.
    Output options are 'formula', or strings.

    Note: currently return FeOT and LOI even for element lists
    not including iron or water - potential upgrade!

    Todo: element verification
    """
    if not elements:
        elements = [el for el in common_elements(output='formula')
                    if not el.__str__() in exclude]
    else:
        # Check that all elements input are indeed elements..
        pass

    oxides = [ox for el in elements
              for ox in simple_oxides(el, output=output)]
    if output != 'formula':
        oxides = [ox.__str__() for ox in oxides] + addition
    return oxides


def simple_oxides(cation, output='formula'):
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
    oxides = [str(cation)+str(1)+'O'+str(c//2)
              if not c%2 else
              str(cation)+str(2)+'O'+str(c)
              for c in ions]
    oxides = [pt.formula(ox) for ox in oxides]
    if not output == 'formula':
        oxides = [ox.__str__() for ox in oxides]
    return oxides


def devolatilise(df: pd.DataFrame,
                 exclude=['H2O', 'H2O_PLUS', 'H2O_MINUS', 'CO2', 'LOI'],
                 renorm=True):
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
    if not (isinstance(oxin, pt.formulas.Formula)
            or isinstance(oxin, pt.core.Element)):
        oxin = pt.formula(oxin)

    if not (isinstance(oxout, pt.formulas.Formula)
            or isinstance(oxout, pt.core.Element)):
        oxout = pt.formula(oxout)

    inatoms = {k: v for (k, v) in oxin.atoms.items() if not k.__str__()=='O'}
    in_els = inatoms.keys()
    outatoms =  {k: v for (k, v) in oxout.atoms.items() if not k.__str__()=='O'}
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
    doc = "Convert series from "+str(oxin)+" to "+str(oxout)
    convert_series.__doc__ = doc
    return convert_series


def recalculate_redox(df: pd.DataFrame,
                      to_oxidised=False,
                      renorm=True,
                      total_suffix='T'):
    """
    Recalculates abundances of redox-sensitive components (particularly Fe),
    and normalises a dataframe to contain only one oxide species for a given
    element.

    Consider reimplementing total suffix as a lambda formatting function
    to deal with cases of prefixes, capitalisation etc.

    Automatic generation of multiple redox species from dataframes
    would also be a natural improvement.
    """
    # Assuming either (a single column) or (FeO + Fe2O3) are reported
    # Fe columns - FeO, Fe2O3, FeOT, Fe2O3T
    FeO = pt.formula("FeO")
    Fe2O3 = pt.formula("Fe2O3")
    dfc = df.copy()
    ox_species = ['Fe2O3', "Fe2O3"+total_suffix]
    ox_in_df = [i for i in ox_species if i in dfc.columns]
    red_species = ['FeO', "FeO"+total_suffix]
    red_in_df = [i for i in red_species if i in dfc.columns]
    if to_oxidised:
        oxFe = oxide_conversion(FeO, Fe2O3)
        Fe2O3T = dfc.loc[:, ox_in_df].fillna(0).sum(axis=1) + \
                 oxFe(dfc.loc[:, red_in_df].fillna(0)).sum(axis=1)
        dfc.loc[:, 'Fe2O3T'] = Fe2O3T
        to_drop = red_in_df + \
                  [i for i in ox_in_df if not i.endswith(total_suffix)]
    else:
        reduceFe = oxide_conversion(Fe2O3, FeO)
        FeOT = dfc.loc[:, red_in_df].fillna(0).sum(axis=1) + \
               reduceFe(dfc.loc[:, ox_in_df].fillna(0)).sum(axis=1)
        dfc.loc[:, 'FeOT'] = FeOT
        to_drop = ox_in_df + \
                  [i for i in red_in_df if not i.endswith(total_suffix)]

    dfc = dfc.drop(columns=to_drop)

    if renorm:
        return renormalise(dfc)
    else:
        return dfc


def aggregate_cation(df: pd.DataFrame,
                     cation,
                     form='oxide',
                     unit_scale=None):
    """
    Aggregates cation information from oxide and elemental components
    to a single series. Allows scaling (e.g. from ppm to wt% - a factor
    of 10,000).

    Needs to also implement a 'molecular' version.
    """
    elstr = cation.__str__()
    oxstr = [o for o in df.columns if o in simple_oxides(elstr, output='str')][0]
    el, ox = pt.formula(elstr), pt.formula(oxstr)

    if form == 'oxide':
        if unit_scale is None: unit_scale = 1/10000 # ppm to Wt%
        assert unit_scale > 0
        convert_function = oxide_conversion(ox, el)
        conv_values = convert_function(df.loc[:, elstr]).values * unit_scale
        totals = np.nansum(np.vstack((df.loc[:, oxstr].values, conv_values)),
                           axis=0)
        totals[np.isclose(totals, 0)] = np.nan
        df.loc[:, oxstr] = totals
        df.drop(columns=[elstr], inplace=True)
        assert elstr not in df.columns

    elif form == 'element':
        if unit_scale is None: unit_scale = 10000 # Wt% to ppm
        assert unit_scale > 0
        convert_function = oxide_conversion(el, ox)
        conv_values = convert_function(df.loc[:, oxstr]).values * unit_scale
        totals = np.nansum(np.vstack((df.loc[:, elstr].values, conv_values)),
                           axis=0)
        totals[np.isclose(totals, 0)] = np.nan
        df.loc[:, elstr] = totals

        df.drop(columns=[oxstr], inplace=True)
        assert oxstr not in df.columns

    return df


def check_multiple_cation_inclusion(df, exclude=['LOI', 'FeOT', 'Fe2O3T']):
    """
    Returns cations which are present in both oxide and elemental form.

    Todo: Options for output (string/formula).
    """
    major_components = [i for i in common_oxides(output='str')
                        if i in df.columns]
    elements_as_majors = [get_cations(oxide)[0] for oxide in major_components
                          if not oxide in exclude]
    elements_as_traces = [c for c in common_elements(output='formula')
                          if c.__str__() in df.columns]
    return set([el for el in elements_as_majors if el in elements_as_traces])


def add_ratio(df: pd.DataFrame,
              ratio:str,
              alias:str='',
              convert=lambda x: x):
    """
    Add a ratio of components A and B, given in the form of string 'A/B'.
    Returned series be assigned an alias name.
    """
    num, den = ratio.split('/')
    assert num in df.columns
    assert den in df.columns
    name = [ratio if not alias else alias][0]
    conv = convert(df.loc[:, [num, den]])
    conv.loc[(conv[den]==0.), den] = np.nan # avoid inf
    df.loc[:, name] = conv.loc[:, num] / conv.loc[:, den]
    return df


def add_MgNo(df: pd.DataFrame,
             molecularIn=False,
             elemental=False,
             components=False):

    if not molecularIn:
        if components:
            # Iron is split into species
            df.loc[:, 'Mg#'] = df['MgO'] / pt.formula('MgO').mass / \
                       (df['MgO'] / pt.formula('MgO').mass + df['FeO'] / pt.formula('FeO').mass)
        else:
            # Total iron is used
            assert 'FeOT' in df.columns
            df.loc[:, 'Mg#'] = df['MgO'] / pt.formula('MgO').mass / \
                       (df['MgO'] / pt.formula('MgO').mass + df['FeOT'] / pt.formula('FeO').mass)
    else:
        if not elemental:
            # Molecular Oxides
            df.loc[:, 'Mg#'] = df['MgO'] / (df['MgO'] + df['FeO'])
        else:
            # Molecular Elemental
            df.loc[:, 'Mg#'] = df['Mg'] / (df['Mg'] + df['Fe'])


def lambda_lnREE(df,
                 norm_to='Chondrite_PON',
                 exclude=['Pm', 'Eu'],
                 params=None,
                 degree=5):
    """
    Calculates lambda coefficients for a given set of REE data, normalised
    to a specific composition. Lambda factors are given for the
    radii vs. ln(REE/NORM) polynomical combination.
    """
    ree = [i for i in REE() if not str(i) in exclude and
           (str(i) in df.columns or i in df.columns)] # no promethium
    radii = np.array(get_radii(ree))

    if params is None:
        params = OP_constants(radii, degree=degree)

    col_indexes = [i for i in df.columns if str(i) not in exclude]

    if isinstance(norm_to, str):
        norm = ReferenceCompositions()[norm_to]
        norm_abund = np.array([norm[str(el)].value for el in ree])
    else: # list, iterable, pd.Index etc
        norm_abund = np.array([i for i in norm])

    assert len(norm_abund) == len(ree)

    labels = [chr(955) + str(d) for d in range(degree)]
    norm_df = df.loc[:, col_indexes] / norm_abund
    norm_df = norm_df.applymap(np.log)
    return pd.DataFrame([lambdas(norm_df.loc[idx, :],
                                 xs=radii,
                                 params=params,
                                 degree=degree)
                         for idx in df.index],
                       columns=labels)

_RADII = {str(k): v for (k, v) in zip(REE(), [1.160, 1.143, 1.126, 1.109,
                                         1.093, 1.079, 1.066, 1.053,
                                         1.040, 1.027, 1.015, 1.004,
                                         0.994, 0.985, 0.977])}
