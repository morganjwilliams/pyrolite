import numpy as np
import pandas as pd
import scipy
import periodictable as pt
from .mindb import get_mineral_group, list_minerals, parse_composition
from ..comp.codata import renormalise, close
from ..geochem.transform import convert_chemistry, to_molecular
from ..util.log import Handle

logger = Handle(__name__)


def unmix(comp, parts, ord=1, det_lim=0.0001):
    """
    From a composition and endmember components, find a set of weights which best
    approximate the composition as a weighted sum of components.

    Parameters
    --------------
    comp : :class:`numpy.ndarray`
        Array of compositions (shape :math:`n_S, n_C`).
    parts : :class:`numpy.ndarray`
        Array of endmembers (shape :math:`n_E, n_C`).
    ord : :class:`int`
        Order of regularization, defaults to L1 for sparsity.
    det_lim : :class:`float`
        Detection limit, below which minor components will be omitted for sparsity.

    Returns
    --------
    :class:`numpy.ndarray`
        Array of endmember modal abundances (shape :math:`n_S, n_E`)
    """
    nsamples, nscomponents = comp.shape
    nparts, ncomponents = parts.shape
    assert nscomponents == ncomponents
    weights = np.ones((nsamples, nparts))
    weights /= weights.sum()
    bounds = np.array([np.zeros(weights.size), np.ones(weights.size)]).T

    def fn(x, comp, parts):
        x = x.reshape(nsamples, nparts)
        return np.linalg.norm(x.dot(parts) - comp, ord=ord)

    res = scipy.optimize.minimize(
        fn,
        weights.flatten(),
        bounds=bounds,
        args=(comp, parts),
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - nsamples},
    )
    byparts = res.x.reshape(weights.shape)
    byparts[(np.isclose(byparts, 0.0, atol=1e-06) | (byparts <= det_lim))] = 0.0
    # if the abundances aren't already molecular, this would be the last point
    # where access access to the composition of the endmembers is guaranteed
    return close(byparts)


def endmember_decompose(
    composition, endmembers=[], drop_zeros=True, molecular=True, ord=1, det_lim=0.0001
):
    """
    Decompose a given mineral composition to given endmembers.

    Parameters
    -----------
    composition : :class:`~pandas.DataFrame` | :class:`~pandas.Series` | :class:`~periodictable.formulas.Formula` | :class:`str`
        Composition to decompose into endmember components.
    endmembers : :class:`str` | :class:`list` | :class:`dict`
        List of endmembers to use for the decomposition.
    drop_zeros : :class:`bool`, :code:`True`
        Whether to omit components with zero estimated abundance.
    molecular : :class:`bool`, :code:`True`
        Whether to *convert* the chemistry to molecular before calculating the
        decomposition.
    ord : :class:`int`
        Order of regularization passed to :func:`unmix`, defaults to L1 for sparsity.
    det_lim : :class:`float`
        Detection limit, below which minor components will be omitted for sparsity.

    Returns
    ---------
    :class:`pandas.DataFrame`
    """
    # parse composition ----------------------------------------------------------------
    assert isinstance(composition, (pd.DataFrame, pd.Series, pt.formulas.Formula, str))
    if not isinstance(
        composition, pd.DataFrame
    ):  # convert to a dataframe representation
        if isinstance(composition, pd.Series):
            # convert to frame
            composition = to_frame(composition)
        elif isinstance(composition, (pt.formulas.Formula, str)):
            formula = composition
            if isinstance(composition, str):
                formula = pt.formula(formula)

    # parse endmember compositions -----------------------------------------------------
    aliases = None
    if not endmembers:
        logger.warning(
            "No endmembers specified, using all minerals. Expect non-uniqueness."
        )
        # try a decomposition with all the minerals; this will be non-unique
        endmembers = list_minerals()

    if isinstance(endmembers, str):  # mineral group
        Y = get_mineral_group(endmembers).set_index("name")
    elif isinstance(endmembers, (list, set, dict, tuple)):
        if isinstance(endmembers, dict):
            aliases, endmembers = list(endmembers.keys()), list(endmembers.values())
        Y = pd.DataFrame(
            [parse_composition(em) for em in endmembers], index=aliases or endmembers
        )
    else:
        raise NotImplementedError("Unknown endmember specification format.")

    # calculate the decomposition ------------------------------------------------------
    X = renormalise(composition, scale=1.0)
    Y = renormalise(
        convert_chemistry(Y, to=composition.columns)
        .loc[:, composition.columns]
        .fillna(0),
        scale=1.0,
    )
    if molecular:
        X, Y = to_molecular(X), to_molecular(Y)
    # optimise decomposition into endmember components
    modal = pd.DataFrame(
        unmix(X.fillna(0).values, Y.fillna(0).values, ord=ord, det_lim=det_lim),
        index=X.index,
        columns=Y.index,
    )
    if drop_zeros:
        modal = modal.loc[:, modal.sum(axis=0) > 0.0]

    modal = renormalise(modal)
    return modal


def LeMatireOxRatio(df, mode="volcanic"):
    """
    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe containing compositions to calibrate against.
    mode : :class:`str`
        Mode for the correction - 'volcanic' or 'plutonic'.

    Returns
    -------
    :class:`pandas.Series`
        Series with oxidation ratios.

    Notes
    ------
    This is a  FeO / (FeO + Fe2O3) mass ratio, not a standar molar ratio
    Fe2+/(Fe2+ + Fe3+) which is more straightfowardly used; data presented
    should be in mass units.

    References
    ----------
    Le Maitre, R. W (1976). Some Problems of the Projection of Chemical Data
    into Mineralogical Classifications.
    Contributions to Mineralogy and Petrology 56, no. 2 (1 January 1976): 181–89.
    https://doi.org/10.1007/BF00399603.
    """
    if mode.lower().startswith("volc"):
        ratio = (
            0.93
            - 0.0042 * df["SiO2"]
            - 0.022 * df.reindex(columns=["Na2O", "K2O"]).sum(axis=1)
        )
    else:
        ratio = (
            0.88
            - 0.0016 * df["SiO2"]
            - 0.027 * df.reindex(columns=["Na2O", "K2O"]).sum(axis=1)
        )
    ratio.name = "FeO/(FeO+Fe2O3)"
    return ratio


def LeMaitre_Fe_correction(df, mode="volcanic"):
    """
    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe containing compositions to correct iron for.
    mode : :class:`str`
        Mode for the correction - 'volcanic' or 'plutonic'.

    Returns
    -------
    :class:`pandas.DataFrame`
        Series with two corrected iron components (FeO, Fe2O3).

    References
    ----------

    Le Maitre, R. W (1976). Some Problems of the Projection of Chemical Data
    into Mineralogical Classifications.
    Contributions to Mineralogy and Petrology 56, no. 2 (1 January 1976): 181–89.
    https://doi.org/10.1007/BF00399603.

    Middlemost, Eric A. K. (1989). Iron Oxidation Ratios, Norms and the Classification of Volcanic Rocks.
    Chemical Geology 77, 1: 19–26. https://doi.org/10.1016/0009-2541(89)90011-9.
    """
    mass_ratios = LeMatireOxRatio(df, mode=mode)  # mass ratios
    # convert mass ratios to mole (Fe) ratios - moles per unit mass for each
    feo_moles = mass_ratios / pt.formula("FeO").mass
    fe203_moles = (1 - mass_ratios) / pt.formula("Fe2O3").mass * 2
    Fe_mole_ratios = feo_moles / (feo_moles + fe203_moles)

    to = {"FeO": Fe_mole_ratios, "Fe2O3": 1 - Fe_mole_ratios}

    return df.reindex(
        columns=["FeO", "Fe2O3", "FeOT", "Fe2O3T"]
    ).pyrochem.convert_chemistry(to=[to])


def CIPW_norm(df, Fe_correction=None, adjust_all=False):
    """
    Standardised calcuation of estimated mineralogy from bulk rock chemistry.
    Takes a dataframe of chemistry & creates a dataframe of estimated mineralogy.

    This is the CIPW norm of Verma et al. (2003).  This version only uses major
    elements.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe containing compositions to transform.
    Fe_correction : :class:`str`
        Iron correction to apply, if any. Will default to 'LeMaitre'.
    adjust_all : :class:`bool`
        Where correcting iron compositions, whether to adjust all iron
        compositions, or only those where singular components are specified.

    Returns
    --------
    :class:`pandas.DataFrame`

    References
    ----------

    Verma, Surendra P., Ignacio S. Torres-Alvarado, and Fernando Velasco-Tapia (2003).
    A Revised CIPW Norm. Swiss Bulletin of Mineralogy and Petrology 83, 2: 197–216.

    Todo
    ----
    * Note whether data needs to be normalised to 1 or 100?
    """
    logger.warning(
        "The current CIPW Norm implmentation is under continuting development, "
        "and does not yet return expected results."
    )

    noncrit = ['CO2', 'SO3', 'F', 'Cl', 'S', 'Ni', 'Co',
                  'Sr', 'Ba', 'Rb', 'Cs', 'Li', 'Zr', 'Cr', 'V']
    columns = (
        ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO"]
        + ["Na2O", "K2O", "P2O5"]
        + noncrit
    )

    # Check that all of the columns we'd like are present
    to_impute = []
    if not set(columns).issubset(set(df.columns.values)):
        to_impute += [c for c in columns if c not in df.columns]
        # raise warning for missing critical columns
        crit_miss = [c for c in to_impute if (c not in noncrit)]
        if crit_miss:
            logger.warning("Required columns missing: {}".format(", ".join(crit_miss)))

    # Reindex columns to be expected and fill missing ones with zeros
    if to_impute:  # Note that we're adding the columns with default values.
        logger.debug("Adding empty (0) columns: {}".format(", ".join(to_impute)))

    ############################################################################
    if Fe_correction is None:  # default to LeMaitre_Fe_correction
        Fe_correction = "LeMaitre"

    if adjust_all:
        fltr = np.ones(df.index.size, dtype="bool")
    else:
        # check where the iron speciation is already specified or there is no iron
        iron_specified = (
            (df.reindex(columns=["FeO", "Fe2O3"]) > 0).sum(axis=1) == 2
        ) | (
            np.isclose(
                df.reindex(columns=["FeO", "Fe2O3", "FeOT", "Fe2O3T"]).sum(axis=1), 0
            )
        )
        fltr = iron_specified

    if Fe_correction.lower().startswith("lemait"):
        df.loc[fltr, ["FeO", "Fe2O3"]] = LeMaitre_Fe_correction(df.loc[fltr, :])
    else:
        raise NotImplementedError(
            "Iron correction {} not recognised.".format(Fe_correction)
        )

    df = df.reindex(columns=columns).fillna(0)


    majors = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO',
       'Na2O', 'K2O', 'P2O5']

    element_AW = {
        'F': 18.9984032,
        'Cl': 35.4527,
        'S': 32.066,
        'Ni': 58.6934,
        'Co': 58.93320,
        'Sr': 87.62,
        'Ba': 137.327,
        'Rb': 85.4678,
        'Cs': 132.90545,
        'Li': 6.941,
        'Zr': 91.224,
        'Cr': 51.9961,
        'V': 50.9415,
        'O': 15.9994
    }

    element_oxide = {
        'F': 'F',
        'Cl': 'Cl',
        'S': 'S',
        'Ni': 'NiO',
        'Co': 'CoO',
        'Sr': 'SrO',
        'Ba': 'BaO',
        'Rb': 'Rb2O',
        'Cs': 'Cs2O',
        'Li': 'Li2O',
        'Zr': 'ZrO2',
        'Cr': 'Cr2O3',
        'V': 'V2O3'
    }

    mineral_molecular_weights = {
        'Q': 60.0843,
        'Z': 183.3031,
        'Ks': 154.2803,
        'An': 278.207276,
        'Ns': 122.0632,
        'Ac': 462.0083,
        'Di': 225.99234699428553,
        'Tn': 196.0625,
        'Hy': 109.82864699428553,
        'Ab': 524.446,
        'Or': 556.6631,
        'Wo': 116.1637,
        'Ol': 159.57299398857106,
        'Pf': 135.9782,
        'Ne': 284.1088,
        'Lc': 436.4945,
        'Cs': 172.2431,
        'Kp': 316.3259,
        'Ap': 328.8691887,
        'Fr': 94.0762,
        'Pr': 135.96640000000002,
        'Cm': 223.83659999999998,
        'Il': 151.7452,
        'Cc': 100.0892,
        'C': 101.9613,
        'Ru': 79.8988,
        'Mt': 231.53860000000003,
        'Hm': 159.6922,
        'Mg-Di': 216.5504,
        'Mg-Hy': 100.3887,
        'Mg-Ol': 140.6931
    }


    # Convert element ppm to Oxides wt%

    trace = ['F', 'Cl', 'S', 'Ni', 'Co',
               'Sr', 'Ba', 'Rb', 'Cs', 'Li', 'Zr', 'Cr', 'V']

    trace_oxides = ['CO2', 'SO3', 'F', 'Cl', 'S', 'NiO', 'CoO',
                    'SrO', 'BaO', 'Rb2O', 'Cs2O', 'Li2O', 'ZrO2', 'Cr2O3', 'V2O3']

    def calculate_oxide_1(oxide_weight, element_AW, ppm):
        return (oxide_weight/element_AW) * ppm * (10**-4)


    def calculate_oxide_2(oxide_weight, element_AW, ppm):
        return (oxide_weight/(2 * element_AW)) * ppm * (10**-4)


    for element in trace:
        if element in ['Ni', 'Co', 'Sr', 'Ba', 'Zr']:
            oxide_name = element_oxide[element]
            df[oxide_name] = calculate_oxide_1(
                                         pt.formula(oxide_name).mass,
                                         element_AW[element],
                                         df[element])
        elif element in ['S', 'Cl', 'F']:
            df[element] = df[element] * (10**-4)
        else:
            oxide_name = element_oxide[element]
            df[oxide_name] = calculate_oxide_2(
                                         pt.formula(oxide_name).mass,
                                         element_AW[element],
                                         df[element])


    # Adjust majors wt% to 100% then adjust again to account for trace components

    # First adjustment
    df['intial_sum'] = df[majors].sum(axis=1)
    adjustment_factor = 100 / df['intial_sum']
    df[majors] = df[majors].mul(adjustment_factor, axis=0)

    # Second adjustment
    df['major_minor_sum'] = df[majors].sum(axis=1) + df[trace_oxides].sum(axis=1)
    adjustment_factor = 100 / df['major_minor_sum']

    df[majors + trace_oxides] = df[majors + trace_oxides].mul(adjustment_factor, axis=0)

    # Rounding to 3 dp

    df[majors + trace_oxides] = df[majors + trace_oxides].round(3)

    # Calculation of some other parameters

    df['Feo/MgO'] = ((2 * pt.formula("FeO").mass / pt.formula("MgO").mass) *
                     df['Fe2O3'] + df['FeO'] / df['MgO'])

    df['SI'] = 100 * df['MgO'] / (df['MgO'] + df['FeO'] + df['Fe2O3'] + df['Na2O'] + df['K2O'])

    df['AR_True'] = (df['Al2O3'] + df['CaO'] + df['Na2O'] + df['K2O']) / (
                df['Al2O3'] + df['CaO'] - df['Na2O'] - df['K2O'])
    df['AR_False'] = (df['Al2O3'] + df['CaO'] + 2 * df['Na2O']) / (df['Al2O3'] + df['CaO'] - 2 * df['Na2O'])

    df['AR'] = np.where(((df['K2O'] / df['Na2O']) >= 1) & ((df['K2O'] / df['Na2O']) <= 2.5) & (df['SiO2'] > 0.5),
                        df['AR_True'], df['AR_False'])

    df['Mg#'] = 100 * df['MgO'] / (df['MgO'] + df['FeO'])

    # Mole Calculations
    for oxide in majors + trace_oxides:
        df['n_' + oxide] = df[oxide] / pt.formula(oxide).mass

    # Combine minor oxides
    df['n_FeO_corr'] = df['n_FeO'] + df['n_MnO'] + df['n_NiO'] + df['n_CoO']
    df['n_CaO_corr'] = df['n_CaO'] + df['n_SrO'] + df['n_BaO']
    df['n_K2O_corr'] = df['n_K2O'] + df['n_Rb2O'] + df['n_Cs2O']
    df['n_Na2O_corr'] = df['n_Na2O'] + df['n_Li2O']
    df['n_Cr2O3_corr'] = df['n_Cr2O3'] + df['n_V2O3']

    # Corrected oxide molecular weight computations
    df['x_MnO'] = df['n_MnO'] / df['n_FeO_corr']
    df['x_FeO'] = df['n_FeO'] / df['n_FeO_corr']

    df['x_NiO'] = df['n_NiO'] / df['n_FeO_corr']
    df['x_CoO'] = df['n_CoO'] / df['n_FeO_corr']

    df['x_SrO'] = df['n_SrO'] / df['n_CaO_corr']
    df['x_BaO'] = df['n_BaO'] / df['n_CaO_corr']
    df['x_CaO'] = df['n_CaO'] / df['n_CaO_corr']

    df['x_Rb2O'] = df['n_Rb2O'] / df['n_K2O_corr']
    df['x_Cs2O'] = df['n_Cs2O'] / df['n_K2O_corr']
    df['x_K2O'] = df['n_K2O'] / df['n_K2O_corr']

    df['x_Li2O'] = df['n_Li2O'] / df['n_Na2O_corr']
    df['x_Na2O'] = df['n_Na2O'] / df['n_Na2O_corr']

    df['x_V2O3'] = df['n_V2O3'] / df['n_Cr2O3_corr']
    df['x_Cr2O3'] = df['n_Cr2O3'] / df['n_Cr2O3_corr']

    df['n_FeO'] = df['n_FeO_corr']
    df['n_CaO'] = df['n_CaO_corr']
    df['n_K2O'] = df['n_K2O_corr']
    df['n_Na2O'] = df['n_Na2O_corr']
    df['n_Cr2O3'] = df['n_Cr2O3_corr']

    # Corrected normative mineral molecular weight computations
    def corr_m_wt(oxide):
        return(df['x_'+ oxide] * pt.formula(oxide).mass)

    df['MW_FeO_corr'] = corr_m_wt('MnO') + corr_m_wt('NiO') + corr_m_wt('CoO') + corr_m_wt('FeO')
    df['MW_CaO_corr'] = corr_m_wt('BaO') + corr_m_wt('SrO') + corr_m_wt('CaO')
    df['MW_K2O_corr'] = corr_m_wt('Rb2O') + corr_m_wt('Cs2O') + corr_m_wt('K2O')
    df['MW_Na2O_corr'] = corr_m_wt('Li2O') + corr_m_wt('Na2O')
    df['MW_Cr2O3_corr'] = corr_m_wt('V2O3') + corr_m_wt('Cr2O3')

    # Corrected molecular weight of Ca, Na and Fe
    df['MW_Ca_corr'] = df['MW_CaO_corr'] - element_AW['O']
    df['MW_Na_corr'] = (df['MW_Na2O_corr'] - element_AW['O']) / 2
    df['MW_Fe_corr'] = df['MW_FeO_corr'] - element_AW['O']

    mineral_molecular_weights['Fe-Hy'] = df['MW_FeO_corr'] + 60.0843
    mineral_molecular_weights['Fe-Ol'] = (2 * df['MW_FeO_corr']) + 60.0843
    mineral_molecular_weights['Mt'] = df['MW_FeO_corr'] + 159.6882
    mineral_molecular_weights['Il'] = df['MW_FeO_corr'] + 79.8658
    mineral_molecular_weights['An'] = df['MW_CaO_corr'] + 222.129876
    mineral_molecular_weights['Mg-Di'] = df['MW_CaO_corr'] + 160.4730
    mineral_molecular_weights['Wo'] = df['MW_CaO_corr'] + 60.0843
    mineral_molecular_weights['Cs'] = 2 * df['MW_CaO_corr'] + 60.0843
    mineral_molecular_weights['Tn'] = df['MW_CaO_corr'] + 139.9501
    mineral_molecular_weights['Pf'] = df['MW_CaO_corr'] + 79.8558
    mineral_molecular_weights['CaF2-Ap'] = 3 * df['MW_CaO_corr'] + (1 / 3) * df['MW_Ca_corr'] + 154.6101241
    mineral_molecular_weights['CaO-Ap'] = (10 / 3) * df['MW_CaO_corr'] + 141.944522
    mineral_molecular_weights['Cc'] = df['MW_CaO_corr'] + 44.0095
    mineral_molecular_weights['Ab'] = df['MW_Na2O_corr'] + 462.467076
    mineral_molecular_weights['Ne'] = df['MW_Na2O_corr'] + 222.129876
    mineral_molecular_weights['Th'] = df['MW_Na2O_corr'] + 80.0642
    mineral_molecular_weights['Nc'] = df['MW_Na2O_corr'] + 44.0095
    mineral_molecular_weights['Ac'] = df['MW_Na2O_corr'] + 400.0254
    mineral_molecular_weights['Ns'] = df['MW_Na2O_corr'] + 60.0843
    mineral_molecular_weights['Or'] = df['MW_K2O_corr'] + 462.467076
    mineral_molecular_weights['Lc'] = df['MW_K2O_corr'] + 342.298476
    mineral_molecular_weights['Kp'] = df['MW_K2O_corr'] + 222.129876
    mineral_molecular_weights['Ks'] = df['MW_K2O_corr'] + 60.0843
    mineral_molecular_weights['Fe-Di'] = df['MW_FeO_corr'] + df['MW_CaO_corr'] + 120.1686
    mineral_molecular_weights['Cm'] = df['MW_FeO_corr'] + df['MW_Cr2O3_corr']
    mineral_molecular_weights['Hl'] = df['MW_Na_corr'] + 35.4527
    mineral_molecular_weights['Fr'] = df['MW_Ca_corr'] + 37.9968064
    mineral_molecular_weights['Pr'] = df['MW_Fe_corr'] + 64.132

    df['Y'] = 0

    ############################################################################
    # Calculate normative components
    ############################################################################

    # Normative Zircon
    df['Z'] = df['n_ZrO2']
    df['Y'] = df['Z']

    # Normative apatite

    df['Ap'] = np.where(
        df['n_CaO'] >= (3 + 1 / 3) * df['n_P2O5'], df['n_P2O5'], df['n_CaO'] / (3 + 1 / 3)
    ).T

    df['n_CaO_'] = np.where(
        df['n_CaO'] >= (3 + 1 / 3) * df['n_P2O5'], df['n_CaO'] - (3 + 1 / 3) * df['Ap'], 0
    ).T

    df['n_P2O5_'] = np.where(
        df['n_CaO'] < (3 + 1 / 3) * df['n_P2O5'], df['n_P2O5'] - df['Ap'], 0).T

    df['n_CaO'] = df['n_CaO_']
    df['n_P2O5'] = df['n_P2O5_']

    df['FREE_P2O5'] = df['n_P2O5']

    # apatite options where F in present

    df['ap_option'] = np.where(
        df['n_F'] >= (2 / 3) * df['Ap'], 2, 3).T

    df['n_F'] = np.where(
        (df['ap_option']) == 2 & (df['n_F'] > 0), df['n_F'] - (2 / 3 * df['Ap']), df['n_F']).T

    df['CaF2-Ap'] = np.where(
        (df['ap_option']) == 3 & (df['n_F'] > 0), df['n_F'] * 1.5, 0).T

    df['CaO-Ap'] = np.where(
        (df['ap_option']) == 3 & (df['n_F'] > 0), df['n_P2O5'] - (1.5 * df['n_F']), 0).T

    df['Ap'] = np.where(
        (df['ap_option']) == 3 & (df['n_F'] > 0), df['CaF2-Ap'] + df['CaO-Ap'], df['Ap']).T

    df['FREEO_12b'] = np.where(df['ap_option'] == 2, 1 / 3 * df['Ap'], 0).T
    df['FREEO_12c'] = np.where(df['ap_option'] == 3, df['n_F'] / 2, 0).T


    # Normative Fluorite
    df['Fr'] = np.where(df['n_CaO'] >= df['n_F'] / 2, df['n_F'] / 2, df['n_CaO']).T

    df['n_CaO'] = np.where(
        df['n_CaO'] >= df['n_F'] / 2, df['n_CaO'] - df['Fr'], 0).T

    df['n_F'] = np.where(
        df['n_CaO'] >= df['n_F'] / 2, df['n_F'], df['n_F'] - (2 * df['Fr'])).T

    df['FREEO_13'] = df['Fr']
    df['FREE_F'] = df['n_F']


    # Normative halite
    df['Hl'] = np.where(
        df['n_Na2O'] >= 2 * df['n_Cl'], df['n_Cl'], df['n_Na2O'] / 2).T

    df['n_Na2O'] = np.where(
        df['n_Na2O'] >= 2 * df['n_Cl'], df['n_Na2O'] - df['Hl'] / 2, 0).T

    df['n_Cl'] = np.where(
        df['n_Na2O'] >= 2 * df['n_Cl'], df['n_Cl'], df['n_Cl'] - df['Hl']).T

    df['FREE_Cl'] = df['n_Cl']
    df['FREEO_14'] = df['Hl'] / 2


    # Normative thenardite
    df['Th'] = np.where(df['n_Na2O'] >= df['SO3'], df['SO3'], df['n_Na2O']).T

    df['n_Na2O'] = np.where(df['n_Na2O'] >= df['SO3'], df['n_Na2O'] - df['Th'], 0).T

    df['n_SO3'] = np.where(df['n_Na2O'] >= df['SO3'], df['n_SO3'], df['n_SO3'] - df['Th']).T

    df['FREE_SO3'] = df['n_SO3']


    # Normative Pyrite
    df['Pr'] = np.where(df['n_FeO'] >= 2 * df['n_S'], df['n_S'] / 2, df['n_FeO']).T

    df['n_FeO'] = np.where(df['n_FeO'] >= 2 * df['n_S'], df['n_FeO'] - df['Pr'], df['n_FeO'] - df['Pr'] * 2).T

    df['FREE_S'] = np.where(df['n_FeO'] >= 2 * df['n_S'], 0, df['n_FeO']).T

    df['n_FeO'] = df['n_FeO'] - df['FREE_S']

    df['FREEO_16'] = df['Pr']

    # Normative sodium carbonate or calcite

    df['Nc'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_CO2'], df['n_Na2O']).T

    df['n_Na2O'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_Na2O'] - df['Nc'], df['n_Na2O']).T

    df['n_CO2'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_CO2'], df['n_CO2'] - df['Nc']).T

    df['Cc'] = np.where(df['n_CaO'] >= df['n_CO2'], df['n_CO2'], df['n_CaO']).T

    df['CaO'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['CaO'] - df['Cc'], df['CaO']).T

    df['CO2'] = np.where(df['n_Na2O'] >= df['n_CO2'], df['n_CO2'], df['n_CO2'] - df['Cc']).T

    df['FREECO2'] = df['n_CO2']


    # Normative Chromite
    df['Cm'] = np.where(df['n_FeO'] >= df['n_Cr2O3'], df['n_Cr2O3'], df['n_FeO']).T

    df['n_FeO'] = np.where(df['n_FeO'] >= df['n_Cr2O3'], df['n_FeO'] - df['Cm'], 0).T
    df['n_Cr2O3'] = np.where(df['n_FeO'] >= df['n_Cr2O3'], df['n_Cr2O3'] - df['Cm'], df['n_Cr2O3']).T

    df['FREE_CR2O3'] = df['Cm']

    # Normative Ilmenite
    df['Il'] = np.where(df['n_FeO'] >= df['n_TiO2'], df['n_TiO2'], df['n_FeO']).T

    df['n_FeO_'] = np.where(df['n_FeO'] >= df['n_TiO2'], df['n_FeO'] - df['Il'], 0).T

    df['n_TiO2_'] = np.where(df['n_FeO'] >= df['n_TiO2'], 0, df['n_TiO2'] - df['Il']).T

    df['n_FeO'] = df['n_FeO_']

    df['n_TiO2'] = df['n_TiO2_']

    # Normative Orthoclase/potasium metasilicate

    df['Or_p'] = np.where(df['n_Al2O3'] >= df['n_K2O'], df['n_K2O'], df['n_Al2O3']).T

    df['n_Al2O3_'] = np.where(df['n_Al2O3'] >= df['n_K2O'], df['n_Al2O3'] - df['Or_p'], 0).T

    df['n_K2O_'] = np.where(df['n_Al2O3'] >= df['n_K2O'], 0, df['n_K2O'] - df['Or_p']).T

    df['Ks'] = df['n_K2O_']

    df['Y'] = np.where(df['n_Al2O3'] >= df['n_K2O'], df['Y'] + (df['Or_p'] * 6),
                       df['Y'] + (df['Or_p'] * 6 + df['Ks'])).T

    df['n_Al2O3'] = df['n_Al2O3_']
    df['n_K2O'] = df['n_K2O_']

    # Normative Albite
    df['Ab_p'] = np.where(df['n_Al2O3'] >= df['n_Na2O'], df['n_Na2O'], df['n_Al2O3']).T

    df['n_Al2O3_'] = np.where(df['n_Al2O3'] >= df['n_Na2O'], df['n_Al2O3'] - df['Ab_p'], 0).T

    df['n_Na2O_'] = np.where(df['n_Al2O3'] >= df['n_Na2O'], 0, df['n_Na2O'] - df['Ab_p']).T

    df['Y'] = df['Y'] + (df['Ab_p'] * 6)

    df['n_Al2O3'] = df['n_Al2O3_']
    df['n_Na2O'] = df['n_Na2O_']

    # Normative Acmite / sodium metasilicate
    df['Ac'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], df['n_Fe2O3'], df['n_Na2O']).T

    df['n_Na2O_'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], df['n_Na2O'] - df['Ac'], 0).T

    df['n_Fe2O3_'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], 0, df['n_Fe2O3'] - df['Ac']).T

    df['Ns'] = df['n_Na2O_']

    df['Y'] = np.where(df['n_Na2O'] >= df['n_Fe2O3'], df['Y'] + (4 * df['Ac'] + df['Ns']), df['Y'] + 4 * df['Ac']).T

    df['n_Na2O'] = df['n_Na2O_']
    df['n_Fe2O3'] = df['n_Fe2O3_']

    # Normative Anorthite / Corundum
    df['An'] = np.where(df['n_Al2O3'] >= df['n_CaO'], df['n_CaO'], df['n_Al2O3']).T

    df['n_Al2O3_'] = np.where(df['n_Al2O3'] >= df['n_CaO'], df['n_Al2O3'] - df['An'], 0).T

    df['n_CaO_'] = np.where(df['n_Al2O3'] >= df['n_CaO'], 0, df['n_CaO'] - df['An']).T

    df['C'] = df['n_Al2O3_']

    df['n_Al2O3'] = df['n_Al2O3_']

    df['n_CaO'] = df['n_CaO_']

    df['Y'] = df['Y'] + 2 * df['An']

    # Normative Sphene / Rutile
    df['Tn_p'] = np.where(df['n_CaO'] >= df['n_TiO2'], df['n_TiO2'], df['n_CaO']).T

    df['n_CaO_'] = np.where(df['n_CaO'] >= df['n_TiO2'], df['n_CaO'] - df['Tn_p'], 0).T

    df['n_TiO2_'] = np.where(df['n_CaO'] >= df['n_TiO2'], 0, df['n_TiO2'] - df['Tn_p']).T

    df['n_CaO'] = df['n_CaO_']
    df['n_TiO2'] = df['n_TiO2_']

    df['Ru'] = df['n_TiO2']

    df['Y'] = df['Y'] + df['Tn_p']

    # Normative Magnetite / Hematite
    df['Mt'] = np.where(df['n_Fe2O3'] >= df['n_FeO'], df['n_FeO'], df['n_Fe2O3']).T

    df['n_Fe2O3_'] = np.where(df['n_Fe2O3'] >= df['n_FeO'], df['n_Fe2O3'] - df['Mt'], 0).T

    df['n_FeO_'] = np.where(df['n_Fe2O3'] >= df['n_FeO'], 0, df['n_FeO'] - df['Mt']).T

    df['n_Fe2O3'] = df['n_Fe2O3_']

    df['n_FeO'] = df['n_FeO_']

    df['Hm'] = df['n_Fe2O3']

    # Subdivision of some normative minerals
    df['n_MgFe_O'] = df['n_MgO'] + df['n_FeO']

    df['MgO_ratio'] = df['n_MgO'] / df['n_MgFe_O']
    df['FeO_ratio'] = df['n_FeO'] / df['n_MgFe_O']

    # Provisional normative dioside, wollastonite / Hypersthene
    df['Di_p'] = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['n_MgFe_O'], df['n_CaO']).T

    df['n_CaO_'] = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['n_CaO'] - df['Di_p'], 0).T

    df['n_MgFe_O_'] = np.where(df['n_CaO'] >= df['n_MgFe_O'], 0, df['n_MgFe_O'] - df['Di_p']).T

    df['Hy_p'] = df['n_MgFe_O_']

    df['Wo_p'] = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['n_CaO_'], 0).T

    df['Y'] = np.where(df['n_CaO'] >= df['n_MgFe_O'], df['Y'] + (2 * df['Di_p'] + df['Wo_p']),
                       df['Y'] + (2 * df['Di_p'] + df['Hy_p'])).T

    df['n_CaO'] = df['n_CaO_']
    df['n_MgFe_O'] = df['n_MgFe_O_']

    # Normative quartz / undersaturated minerals
    df['Q'] = np.where(df['n_SiO2'] >= df['Y'], df['n_SiO2'] - df['Y'], 0).T

    df['D'] = np.where(df['n_SiO2'] < df['Y'], df['Y'] - df['n_SiO2'], 0).T

    df['deficit'] = df['D'] > 0

    # Normative Olivine / Hypersthene
    df['Ol_'] = np.where((df['D'] < df['Hy_p'] / 2), df['D'], df['Hy_p'] / 2).T

    df['Hy'] = np.where((df['D'] < df['Hy_p'] / 2), df['Hy_p'] - 2 * df['D'], 0).T

    df['D1'] = df['D'] - df['Hy_p'] / 2

    df['Ol'] = np.where((df['deficit']), df['Ol_'], 0).T

    df['Hy'] = np.where((df['deficit']), df['Hy'], df['Hy_p']).T

    df['deficit'] = df['D1'] > 0

    # Normative Sphene / Perovskite
    df['Tn'] = np.where((df['D1'] < df['Tn_p']), df['Tn_p'] - df['D1'], 0).T

    df['Pf_'] = np.where((df['D1'] < df['Tn_p']), df['D1'], df['Tn_p']).T

    df['D2'] = df['D1'] - df['Tn_p']

    df['Tn'] = np.where((df['deficit']), df['Tn'], df['Tn_p']).T
    df['Pf'] = np.where((df['deficit']), df['Pf_'], 0).T

    df['deficit'] = df['D2'] > 0

    # Normative Nepheline / Albite
    df['Ne_'] = np.where((df['D2'] < 4 * df['Ab_p']), df['D2'] / 4, df['Ab_p']).T

    df['Ab'] = np.where((df['D2'] < 4 * df['Ab_p']), df['Ab_p'] - df['D2'] / 4, 0).T

    df['D3'] = df['D2'] - 4 * df['Ab_p']

    df['Ne'] = np.where((df['deficit']), df['Ne_'], 0).T
    df['Ab'] = np.where((df['deficit']), df['Ab'], df['Ab_p']).T

    df['deficit'] = df['D3'] > 0

    # Normative Leucite / Orthoclase
    df['Lc'] = np.where((df['D3'] < 2 * df['Or_p']), df['D3'] / 2, df['Or_p']).T

    df['Or'] = np.where((df['D3'] < 2 * df['Or_p']), df['Or_p'] - df['D3'] / 2, 0).T

    df['D4'] = df['D3'] - 2 * df['Or_p']

    df['Lc'] = np.where((df['deficit']), df['Lc'], 0).T
    df['Or'] = np.where((df['deficit']), df['Or'], df['Or_p']).T

    df['deficit'] = df['D4'] > 0

    # Normative dicalcium silicate / wollastonite
    df['Cs'] = np.where((df['D4'] < df['Wo_p'] / 2), df['D4'], df['Wo_p'] / 2).T

    df['Wo'] = np.where((df['D4'] < df['Wo_p'] / 2), df['Wo_p'] - 2 * df['D4'], 0).T

    df['D5'] = df['D4'] - df['Wo_p'] / 2

    df['Cs'] = np.where((df['deficit']), df['Cs'], 0).T
    df['Wo'] = np.where((df['deficit']), df['Wo'], df['Wo_p']).T

    df['deficit'] = df['D5'] > 0

    # Normative dicalcium silicate / Olivine Adjustment
    df['Cs_'] = np.where((df['D5'] < df['Di_p']), df['D5'] / 2 + df['Cs'], df['Di_p'] / 2 + df['Cs']).T

    df['Ol_'] = np.where((df['D5'] < df['Di_p']), df['D5'] / 2 + df['Ol'], df['Di_p'] / 2 + df['Ol']).T

    df['Di_'] = np.where((df['D5'] < df['Di_p']), df['Di_p'] - df['D5'], 0).T

    df['D6'] = df['D5'] - df['Di_p']

    df['Cs'] = np.where((df['deficit']), df['Cs_'], df['Cs']).T
    df['Ol'] = np.where((df['deficit']), df['Ol_'], df['Ol']).T
    df['Di'] = np.where((df['deficit']), df['Di_'], df['Di_p']).T

    df['deficit'] = df['D6'] > 0

    # Normative Kaliophilite / Leucite
    df['Kp'] = np.where((df['Lc'] >= df['D6'] / 2), df['D6'] / 2, df['Lc']).T

    df['Lc_'] = np.where((df['Lc'] >= df['D6'] / 2), df['Lc'] - df['D6'] / 2, 0).T

    df['Kp'] = np.where((df['deficit']), df['Kp'], 0).T
    df['Lc'] = np.where((df['deficit']), df['Lc_'], df['Lc']).T

    df['DEFSIO2'] = np.where((df['Lc'] < df['D6'] / 2) & (df['deficit']), df['D6'] - 2 * df['Kp'], 0).T

    # Allocate definite mineral proportions
    ## Subdivide Hypersthene, Diopside and Olivine into Mg- and Fe- varieties

    df['Fe-Hy'] = df['Hy'] * df['FeO_ratio']
    df['Fe-Di'] = df['Di'] * df['FeO_ratio']
    df['Fe-Ol'] = df['Ol'] * df['FeO_ratio']

    df['Mg-Hy'] = df['Hy'] * df['MgO_ratio']
    df['Mg-Di'] = df['Di'] * df['MgO_ratio']
    df['Mg-Ol'] = df['Ol'] * df['MgO_ratio']

    mineral_proportions = pd.DataFrame()
    mineral_pct_mm = pd.DataFrame()
    FREE = pd.DataFrame()

    FREE['FREEO_12b'] = (
                                1 + ((0.1) * ((mineral_molecular_weights['CaF2-Ap'] / 328.86918) - 1))
                        ) * element_AW['O'] * df['FREEO_12b']
    FREE['FREEO_12c'] = (
                                1 + ((0.1) * (df['CaF2-Ap'] / df['Ap']) * (
                                    (mineral_molecular_weights['CaF2-Ap'] / 328.86918) - 1))
                        ) * element_AW['O'] * df['FREEO_12c']

    FREE['FREEO_13'] = (
                               1 + ((pt.formula('CaO').mass // 56.0774) - 1)
                       ) * element_AW['O'] * df['FREEO_13']

    FREE['FREEO_14'] = (
                               1 + (0.5 * ((pt.formula('Na2O').mass / 61.9789) - 1))
                       ) * element_AW['O'] * df['FREEO_14']

    FREE['FREEO_16'] = (
                               1 + ((pt.formula('FeO').mass // 71.8444) - 1)
                       ) * element_AW['O'] * df['FREEO_16']

    FREE['O'] = FREE[['FREEO_12b', 'FREEO_12c', 'FREEO_13', 'FREEO_14', 'FREEO_16']].sum(axis=1)

    FREE['CO2'] = df['FREECO2'] * 44.0095

    FREE['P2O5'] = df['FREE_P2O5'] * 141.94452
    FREE['F'] = df['FREE_F'] * 18.9984032
    FREE['Cl'] = df['FREE_Cl'] * 35.4527
    FREE['SO3'] = df['FREE_SO3'] * 80.0642
    FREE['S'] = df['FREE_S'] * 32.066
    FREE['Cr2O3'] = df['FREE_CR2O3'] * 151.990

    FREE['OXIDES'] = FREE[['P2O5', 'F', 'Cl', 'SO3', 'S', 'Cr2O3']].sum(axis=1)
    FREE['DEFSIO2'] = df['DEFSIO2'] * 60.0843
    FREE.drop(['P2O5', 'F', 'Cl', 'SO3', 'S', 'Cr2O3'], axis=1, inplace=True)

    not_subdivided = ['Hy', 'Di', 'Ol']
    mineral_codes = {
        'Q': 'Quartz',
        'Z': 'Zircon',
        'Ks': 'K2SiO3',
        'An': 'Anorthite',
        'Ns': 'Na2SiO3',
        'Ac': 'Acmite',
        'Di': 'Diopside',
        'Fe-Di': 'Clinoferrosilite',
        'Mg-Di': 'Clinoentatite',
        'Tn': 'Sphene',
        'Hy': 'Hypersthene',
        'Fe-Hy': 'Ferrosilite',
        'Mg-Hy': 'Enstatite',
        'Ab': 'Albite',
        'Or': 'Orthoclase',
        'Wo': 'Wollastonite',
        'Ol': 'Olivine',
        'Fe-Ol': 'Fayalite',
        'Mg-Ol': 'Forsterite',
        'Pf': 'Perovskite',
        'Ne': 'Nepheline',
        'Lc': 'Leucite',
        'Cs': 'Larnite',
        'Kp': 'Kalsilite',
        'Ap': 'Apatite',
        'Fr': 'Fluorite',
        'Pr': 'Pyrite',
        'Cm': 'Chromite',
        'Il': 'Ilmenite',
        'Cc': 'Calcite',
        'C': 'Corundum',
        'Ru': 'Rutile',
        'Mt': 'Magnetite',
        'Hm': 'Hematite'
    }

    for mineral in mineral_codes:
        if mineral in not_subdivided:
            pass
        elif mineral == ['Ap']:
            mineral_pct_mm[mineral] = np.where(
                df['ap_option'] == 2, df[mineral] * mineral_molecular_weights['CaO-Ap'],
                (df['CaF2-Ap'] * mineral_molecular_weights['CaF2-Ap']) + (
                        df['CaO-Ap'] * mineral_molecular_weights['CaO-Ap']))
        else:
            mineral_proportions[mineral] = df[mineral]
            mineral_pct_mm[mineral] = mineral_proportions[mineral] * mineral_molecular_weights[mineral]

    mineral_pct_mm.rename(columns=mineral_codes, inplace=True)
    mineral_pct_mm.fillna(0, inplace=True)

    return(mineral_pct_mm)