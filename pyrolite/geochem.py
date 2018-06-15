import pandas as pd
import numpy as np
import mpmath
import periodictable as pt
from .compositions import renormalise
from .text_utilities import titlecase
import matplotlib.pyplot as plt
import ternary
import warnings

def to_molecular(df: pd.DataFrame, renorm=True):
    """
    Converts mass quantities to molar quantities of the same order.
    E.g.:
    mass% --> mol%
    mass-ppm --> mol-ppm
    """
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


def REE_elements(output='formula', include_extras=False):
    """
    Provides the list of Rare Earth Elements
    Output options are 'formula', or strings.

    Todo: add include extras such as Y.
    """
    elements = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    if output == 'formula':
        elements = [getattr(pt, el) for el in elements]
        #elements = [el.__str__() for el in elements]
    return elements


def common_oxides(elements: list=[], output='formula',
                  addition: list=['FeOT', 'Fe2O3T', 'LOI']):
    """
    Creates a list of oxides based on a list of elements.
    Output options are 'formula', or strings.

    Note: currently return FeOT and LOI even for element lists
    not including iron or water - potential upgrade!

    Todo: element verification
    """
    if not elements:
        elements = [el for el in common_elements(output='formula')
                    if not el.__str__() == 'O']  # Exclude oxygen
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
    oxides = [pt.formula(f'{cation}{1}O{c//2}') if not c%2
              else pt.formula(f'{cation}{2}O{c}') for c in ions]
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
    doc = f"""Convert series from {str(oxin)} to {str(oxout)}"""
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
    ox_species = ['Fe2O3', f"Fe2O3{total_suffix}"]
    ox_in_df = [i for i in ox_species if i in dfc.columns]
    red_species = ['FeO', f"FeO{total_suffix}"]
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
    Aggregates cation information from oxide and elemental components to a single series.
    Allows scaling (e.g. from ppm to wt% - a factor of 10,000).

    Needs to also implement a 'molecular' version.
    """
    elstr = cation.__str__()
    oxstr = [o for o in df.columns if o in simple_oxides(elstr, output='str')][0]
    el, ox = pt.formula(elstr), pt.formula(oxstr)

    if form == 'oxide':
        if unit_scale is None: unit_scale = 1/10000 # ppm to Wt%
        convert_function = oxide_conversion(ox, el)
        conv_values = convert_function(df.loc[:, elstr]).values * unit_scale
        df.loc[:, oxstr] = np.nansum(np.vstack((df.loc[:, oxstr].values, conv_values)), axis=0)
        df = df.loc[:, [i for i in df.columns if not i == elstr]]
    elif form == 'element':
        if unit_scale is None: unit_scale = 10000 # Wt% to ppm
        convert_function = oxide_conversion(el, ox)
        conv_values = convert_function(df.loc[:, oxstr]).values * unit_scale
        df.loc[:, elstr] += np.nansum(np.vstack((df.loc[:, elstr].values, conv_values)), axis=0)
        df = df.loc[:, [i for i in df.columns if not i == oxstr]]

    return df


def check_multiple_cation_inclusion(df, exclude=['LOI', 'FeOT', 'Fe2O3T']):
    major_components = [i for i in common_oxides(output='str') if i in df.columns]
    elements_as_majors = [get_cations(oxide)[0] for oxide in major_components if not oxide in exclude]
    elements_as_traces = [c for c in common_elements(output='formula') if c.__str__() in df.columns]
    return [el for el in elements_as_majors if el in elements_as_traces]


def add_ratio(df: pd.DataFrame,
              ratio:str,
              alias:str='',
              convert=lambda x: x):
    """
    Add a ratio of components A and B, given in the form of string 'A/B'.
    Returned series be assigned an alias name.
    """
    num, den = ratio.split('/')
    name = [ratio if not alias else alias][0]
    conv = convert(df.loc[:, [num, den]])
    conv.loc[(conv[den]==0.), den] = np.nan # avoid inf
    df.loc[:, name] = conv.loc[:, num] / conv.loc[:, den]


def add_MgNo(df: pd.DataFrame,
             molecularIn=False,
             elemental=False,
             components=False):

    if not molecularIn:
        if components:
            # Iron is split into species
            df['Mg#'] = df['MgO'] / pt.formula('MgO').mass / \
                       (df['MgO'] / pt.formula('MgO').mass + df['FeO'] / pt.formula('FeO').mass)
        else:
            # Total iron is used
            assert 'FeOT' in df.columns
            df['Mg#'] = df['MgO'] / pt.formula('MgO').mass / \
                       (df['MgO'] / pt.formula('MgO').mass + df['FeOT'] / pt.formula('FeO').mass)
    else:
        if not elemental:
            # Molecular Oxides
            df['Mg#'] = df['MgO'] / (df['MgO'] + df['FeO'])
        else:
            # Molecular Elemental
            df['Mg#'] = df['Mg'] / (df['Mg'] + df['Fe'])



def lambdas(REE, degrees=2, constructor=mpmath.chebyu):
    """
    Defaults to the  Chebyshev polynomials of the second kind.
    """
    lambs = [lambda x: constructor(deg, x) for deg in range(degrees)]
    print(lambs)
    mpmath.plot(lambs,[-1,1])


def spiderplot(df, ax=None, components:list=None, plot=True, fill=False, **style):
    """
    Plots spidergrams for trace elements data.
    By using separate lines and scatterplots, values between two null-valued
    items are still presented. Might be able to speed up the lines
    with a matplotlib.collections.LineCollection

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    components: list, None
        Elements or compositional components to plot.
    plot: boolean, True
        Whether to plot lines and markers.
    fill:
        Whether to add a patch representing the full range.
    style:
        Styling keyword arguments to pass to matplotlib.
    """

    try:
        assert plot or fill
    except:
        raise AssertionError('Please select to either plot values or fill between ranges.')
    sty = {}
    # Some default values
    sty['marker'] = style.get('marker') or 'D'
    sty['color'] = style.get('color') or style.get('c') or None
    sty['alpha'] = style.get('alpha') or style.get('a') or 1.
    if sty['color'] is None:
        del sty['color']

    components = components or [el for el in common_elements(output='str')
                                if el in df.columns]
    assert len(components) != 0
    c_indexes = np.arange(len(components))

    ax = ax or plt.subplots(1, figsize=(len(components)*0.25, 4))[1]

    if plot:
        ls = ax.plot(c_indexes,
                     df[components].T.values.astype(np.float),
                     **sty)

        sty['s'] = style.get('markersize') or style.get('s') or 5.
        if sty.get('color') is None:
            sty['color'] = ls[0].get_color()
        sc = ax.scatter(np.tile(c_indexes, (df[components].index.size,1)).T,
                        df[components].T.values.astype(np.float), **sty)

    for s_item in ['marker', 's']:
        if s_item in sty:
            del sty[s_item]

    if fill:
        mins, maxs = df[components].min(axis=0), df[components].max(axis=0)
        ax.fill_between(c_indexes, mins, maxs, **sty)

    ax.set_xticks(c_indexes)
    ax.set_xticklabels(components, rotation=60)
    ax.set_yscale('log')
    ax.set_xlabel('Element')

    unused_keys = [i for i in style if i not in list(sty.keys()) + \
                  ['alpha', 'a', 'c', 'color', 'marker']]
    if len(unused_keys):
        warnings.warn(f'Styling not yet implemented for :{unused_keys}')


def ternaryplot(df, ax=None, components=None, **kwargs):
    """
    Plots scatter ternary diagrams, using a wrapper around the
    python-ternary library (gh.com/marcharper/python-ternary).

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe from which to draw data.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    components: list, None
        Elements or compositional components to plot.
    """

    try:
        assert (len(df.columns)==3) or (len(components)==3)
        components = components or df.columns.values
    except:
        raise AssertionError('Please either suggest three elements or a 3-element dataframe.')

    # Some default values
    scale = kwargs.get('scale') or 100.
    figsize = kwargs.get('size') or 8.
    gridsize = kwargs.get('gridsize') or 10.
    fontsize = kwargs.get('fontsize') or 12.

    sty = {}
    sty['marker'] = kwargs.get('marker') or 'D'
    sty['color'] = kwargs.get('color') or kwargs.get('c') or '0.5'
    sty['label'] = kwargs.get('label') or None
    sty['alpha'] = kwargs.get('alpha') or kwargs.get('a') or 1.

    ax = ax or plt.subplots(1, figsize=(figsize, figsize* 3**0.5 * 0.5))[1]
    d1 = ax.__dict__.copy()
    tax = getattr(ax, 'tax', None) or ternary.figure(ax=ax, scale=scale)[1]
    ax.tax = tax
    points = df.loc[:, components].div(df.loc[:, components].sum(axis=1), axis=0).values * scale
    sc = tax.scatter(points, **sty)

    if sty['label'] is not None:
        tax.legend(frameon=False,)

    if not len(tax._labels.keys()): # Checking if there's already a ternary axis
        tax.left_axis_label(components[2], fontsize=fontsize)
        tax.bottom_axis_label(components[0], fontsize=fontsize)
        tax.right_axis_label(components[1], fontsize=fontsize)

        tax.gridlines(multiple=gridsize, color='k', alpha=0.5)
        tax.ticks(axis='lbr', linewidth=1, multiple=gridsize)
        tax.boundary(linewidth=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return tax
