import re
import warnings

import numpy as np
import pandas as pd
import periodictable as pt
import scipy

from ..comp.codata import close, renormalise
from ..geochem.transform import convert_chemistry, to_molecular
from ..util.log import Handle
from ..util.units import scale
from .mindb import get_mineral_group, list_minerals, parse_composition

logger = Handle(__name__)

# minerals for the CIPW Norm
NORM_MINERALS = {
    "Q": {"name": "quartz", "formulae": "SiO2"},
    "Z": {"name": "zircon", "formulae": "ZrO2 SiO2", "SINCLAS_abbrv": "ZIR"},
    "Ks": {"name": "potassium metasilicate", "formulae": "K2O SiO2"},
    "An": {"name": "anorthite", "formulae": "CaO Al2O3 (SiO2)2"},
    "Ns": {"name": "sodium metasilicate", "formulae": "Na2O SiO2"},
    "Ac": {"name": "acmite", "formulae": "Na2O Fe2O3 (SiO2)4"},
    "Th": {"name": "thenardite", "formulae": "Na2O SO3"},
    "Ab": {"name": "albite", "formulae": "Na2O Al2O3 (SiO2)6"},
    "Or": {"name": "orthoclase", "formulae": "K2O Al2O3 (SiO2)6"},
    "Pf": {"name": "perovskite", "formulae": "CaO TiO2", "SINCLAS_abbrv": "PER"},
    "Ne": {"name": "nepheline", "formulae": "Na2O Al2O3 (SiO2)2"},
    "Lc": {"name": "leucite", "formulae": "K2O Al2O3 (SiO2)4"},
    "Cs": {"name": "dicalcium silicate", "formulae": "(CaO)2 SiO2"},
    "Kp": {"name": "kaliophilite", "formulae": "K2O Al2O3 (SiO2)2"},
    "Ap": {"name": "apatite", "formulae": "(CaO)3 P2O5 (CaO)0.33333"},
    "CaF2-Ap": {"name": "fluroapatite", "formulae": "(CaO)3 P2O5 (CaO)0.33333"},
    "Fr": {"name": "fluorite", "formulae": "CaF2"},
    "Pr": {"name": "pyrite", "formulae": "FeS2", "SINCLAS_abbrv": "PYR"},
    "Cm": {"name": "chromite", "formulae": "FeO Cr2O3", "SINCLAS_abbrv": "CHR"},
    "Il": {"name": "ilmenite", "formulae": "FeOTiO2"},
    "Cc": {"name": "calcite", "formulae": "CaO CO2"},
    "C": {"name": "corundum", "formulae": "Al2O3"},
    "Ru": {"name": "rutile", "formulae": "TiO2"},
    "Mt": {"name": "magnetite", "formulae": "FeO Fe2O3"},
    "Hm": {"name": "hematite", "formulae": "Fe2O3", "SINCLAsS_abbrv": "HE"},
    "Mg-Ol": {"name": "forsterite", "formulae": "(MgO)2 SiO2", "SINCLAS_abbrv": "FO"},
    "Fe-Ol": {"name": "fayalite", "formulae": "(FeO)2 SiO2", "SINCLAS_abbrv": "FA"},
    "Fe-Di": {
        "name": "clinoferrosilite",
        "formulae": "CaO FeO (SiO2)2",
        "SINCLAS_abbrv": "DIF",
    },
    "Mg-Di": {
        "name": "clinoenstatite",
        "formulae": "CaO MgO (SiO2)2",
        "SINCLAS_abbrv": "DIM",
    },
    "Fe-Hy": {"name": "ferrosilite", "formulae": "FeO SiO2", "SINCLAS_abbrv": "HYF"},
    "Mg-Hy": {"name": "enstatite", "formulae": "MgO SiO2", "SINCLAS_abbrv": "HYM"},
    "Wo": {"name": "wollastonite", "formulae": "CaO SiO2"},
    "Nc": {"name": "cancrinite", "formulae": "Na2O CO2", "SINCLAS_abbrv": "CAN"},
    "Hl": {"name": "halite", "formulae": "NaCl", "SINCLAS_abbrv": "HL"},
    "Tn": {"name": "titanite", "formulae": "CaO TiO2 SiO2", "SINCLAS_abbrv": "SPH"},
    "Di": {"name": "diopside", "formulae": "CaO MgO (SiO2)2"},
    "Hy": {"name": "hypersthene", "formulae": "MgO SiO2"},
    "Ol": {"name": "olivine", "formulae": "(MgO)2 SiO2"},
}

# Add standard masses to minerals
for mineral in NORM_MINERALS.keys():
    NORM_MINERALS[mineral]["mass"] = pt.formula(NORM_MINERALS[mineral]["formulae"]).mass


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


################################################################################
# CIPW Norm and Related functions
################################################################################


def LeMaitreOxRatio(df, mode=None):
    r"""
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
    This is a  :math:`\mathrm{FeO / (FeO + Fe_2O_3)}` mass ratio, not a standar
    molar ratio  :math:`\mathrm{Fe^{2+}/(Fe^{2+} + Fe^{3+})}` which is more
    straightfowardly used; data presented should be in mass units. For the
    calculation, SiO2, Na2O and K2O are expected to be present.
    References
    ----------
    Le Maitre, R. W (1976). Some Problems of the Projection of Chemical Data
    into Mineralogical Classifications.
    Contributions to Mineralogy and Petrology 56, no. 2 (1 January 1976): 181–89.
    https://doi.org/10.1007/BF00399603.
    """
    if mode is None:  # defualt to volcanic
        mode = "volcanic"

    missing_columns = [c for c in ["SiO2", "Na2O", "K2O"] if c not in df.columns]
    if missing_columns:
        logger.warning(
            "Missing columns required for calculation of iron oxidation"
            "speciation ratio: {}.".format(",".join(missing_columns))
        )

    if mode.lower().startswith("volc"):
        logger.debug("Using LeMaitre Volcanic Fe Correction.")
        ratio = (
            0.93
            - 0.0042 * df["SiO2"]
            - 0.022 * df.reindex(columns=["Na2O", "K2O"]).sum(axis=1)
        )
    else:
        logger.debug("Using LeMaitre Plutonic Fe Correction.")
        ratio = (
            0.88
            - 0.0016 * df["SiO2"]
            - 0.027 * df.reindex(columns=["Na2O", "K2O"]).sum(axis=1)
        )
    ratio.name = "FeO/(FeO+Fe2O3)"
    return ratio


def LeMaitre_Fe_correction(df, mode="volcanic"):
    r"""
    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe containing compositions to correct iron for.
    mode : :class:`str`
        Mode for the correction - 'volcanic' or 'plutonic'.
    Returns
    -------
    :class:`pandas.DataFrame`
        Series with two corrected iron components
        (:math:`\mathrm{FeO, Fe_2O_3}`).
    References
    ----------
    Le Maitre, R. W (1976). Some Problems of the Projection of Chemical Data
    into Mineralogical Classifications.
    Contributions to Mineralogy and Petrology 56, no. 2 (1 January 1976): 181–89.
    https://doi.org/10.1007/BF00399603.
    Middlemost, Eric A. K. (1989). Iron Oxidation Ratios, Norms and the
    Classification of Volcanic Rocks. Chemical Geology 77, 1: 19–26.
    https://doi.org/10.1016/0009-2541(89)90011-9.
    """
    mass_ratios = LeMaitreOxRatio(df, mode=mode)  # mass ratios
    # convert mass ratios to mole (Fe) ratios - moles per unit mass for each
    feo_moles = mass_ratios / pt.formula("FeO").mass
    fe203_moles = (1 - mass_ratios) / pt.formula("Fe2O3").mass * 2
    Fe_mole_ratios = feo_moles / (feo_moles + fe203_moles)

    to = {"FeO": Fe_mole_ratios, "Fe2O3": 1 - Fe_mole_ratios}

    return df.reindex(
        columns=["FeO", "Fe2O3", "FeOT", "Fe2O3T"]
    ).pyrochem.convert_chemistry(to=[to])


def _update_molecular_masses(mineral_dict, corrected_mass_df):
    """
    Update a dictionary of mineral molecular masses based on their oxide
    components. Note that this modifies in place and has no return value.
    Parameters
    ----------
    mineral_dict : :class:`dict`
        Dictionary of minerals containing compositions and molecular masses.
    corrected_mass_df : :class:`dict`
        Dataframe containing columns which include corrected molecular masses
        for specific oxide components.
    """
    for mineral, data in mineral_dict.items():
        composition = data["formulae"]
        masses = 0.0
        for oxide in composition.split():
            count = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", oxide)
            normalised_oxide_name = re.findall(r"[a-zA-Z\d]+", oxide)[0]

            if len(count) > 0:
                count = float(count[0])
            else:
                count = 1

            if normalised_oxide_name in corrected_mass_df:
                mass = count * corrected_mass_df[normalised_oxide_name]
            else:
                # get the components which don't have adjusted molecular weights
                mass = pt.formula(normalised_oxide_name).mass * count
            masses += mass
        data["mass"] = masses


def _aggregate_components(df, to_component, from_components, corrected_mass):
    """
    Aggregate minor components into major oxides and cacluate associated
    minor component fractions and a corrected molecular weight for the major
    oxide component. Note that this modifies in place and has no return value.
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe to aggregate molar components from.
    to_component : :class:`str`
        Major oxide component to aggreagte to.
    from_components : :class:`list`
        Minor oxide components to aggregate from.
    corrected_mass : :class:`pandas.DataFrame`
        Dataframe to put corrected masses.
    """
    target = "n_{}_corr".format(to_component)
    # ensure the main component is included..
    from_components = list(set([to_component] + from_components))
    n_components = ["{}".format(f) for f in from_components]
    x_components = ["x_{}".format(f) for f in from_components]
    df[target] = df[n_components].sum(axis=1)
    logger.debug("Aggregating {} to {}.".format(",".join(n_components), target))
    df[x_components] = df[n_components].div(df[target], axis=0)
    corrected_mass[to_component] = df[x_components] @ np.array(
        [pt.formula(f.replace("n_", "")).mass for f in from_components]
    )


def CIPW_norm(df, Fe_correction=None, Fe_correction_mode=None, adjust_all_Fe=False):
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
    Fe_correction_mode : :class:`str`
        Mode for the iron correction, where applicable.
    adjust_all_Fe : :class:`bool`
        Where correcting iron compositions, whether to adjust all iron
        compositions, or only those where singular components are specified.
    Returns
    --------
    :class:`pandas.DataFrame`
    References
    ----------
    Verma, Surendra P., Ignacio S. Torres-Alvarado, and Fernando Velasco-Tapia (2003).
    A Revised CIPW Norm. Swiss Bulletin of Mineralogy and Petrology 83, 2: 197–216.
    Verma, S. P., & Rivera-Gomez, M. A. (2013). Computer Programs for the
    Classification and Nomenclature of Igneous Rocks. Episodes, 36(2), 115–124.
    Todo
    ----
    * Note whether data needs to be normalised to 1 or 100?
    Notes
    -----
    The function expect oxide components to be in wt% and elemental data to be
    in ppm.
    """
    warnings.warn(
        "The current CIPW Norm implmentation is under continuting development, "
        "and does not yet return expected results."
    )
    minerals = {**NORM_MINERALS}  # copy of NORM_MINERALS
    noncrit = [
        "CO2",
        "SO3",
        "F",
        "Cl",
        "S",
        "Ni",
        "Co",
        "Sr",
        "Ba",
        "Rb",
        "Cs",
        "Li",
        "Zr",
        "Cr",
        "V",
    ]
    columns = (
        ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO"]
        + ["Na2O", "K2O", "P2O5"]
        + noncrit
    )

    majors = [
        "SiO2",
        "TiO2",
        "Al2O3",
        "Fe2O3",
        "FeO",
        "MnO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "P2O5",
    ]

    trace = ["F", "Cl", "S", "Ni", "Co", "Sr", "Ba", "Rb", "Cs", "Li", "Zr", "Cr", "V"]

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

    if adjust_all_Fe:
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
        df.loc[fltr, ["FeO", "Fe2O3"]] = LeMaitre_Fe_correction(
            df.loc[fltr, :], mode=Fe_correction_mode
        )
    else:
        raise NotImplementedError(
            "Iron correction {} not recognised.".format(Fe_correction)
        )

    df = df.reindex(columns=columns).fillna(0)

    # convert ppm traces to wt%
    df.loc[:, trace] *= scale("ppm", "wt%")

    # define the form which we want our minor and trace components to be in
    minors_trace = [
        "F",
        "Cl",
        "S",
        "CO2",
        "NiO",
        "CoO",
        "SrO",
        "BaO",
        "Rb2O",
        "Cs2O",
        "Li2O",
        "ZrO2",
        "Cr2O3",
        "V2O3",
    ]

    SO3 = df["SO3"]

    # convert to a single set of oxides and gas traces
    df = df.pyrochem.convert_chemistry(to=majors + minors_trace, renorm=False).fillna(0)

    df["SO3"] = SO3

    ############################################################################
    # Normalization
    # Adjust majors wt% to 100% then adjust again to account for trace components
    ############################################################################
    # Rounding to 3 dp
    df[majors] = df[majors].round(3)

    # First adjustment
    df["intial_sum"] = df[majors].sum(axis=1)
    adjustment_factor = 100.0 / df["intial_sum"]
    df[majors] = df[majors].mul(adjustment_factor, axis=0)

    # Second adjustment
    df["major_minor_sum"] = df[majors].sum(axis=1) + df[minors_trace].sum(axis=1)
    adjustment_factor = 100.0 / df["major_minor_sum"]

    df[majors + minors_trace] = df[majors + minors_trace].mul(adjustment_factor, axis=0)

    ox_adjusted = df.copy(deep=True)

    ############################################################################
    # Mole Calculations
    # TODO: update to use df.pyrochem.to_molecular()
    ############################################################################
    for component in majors + minors_trace:
        df[component] = df[component] / pt.formula(component).mass

    ############################################################################
    # Combine minor components, compute minor component fractions and correct masses
    ############################################################################

    ###########################
    # Code below does not work
    ###########################

    # corrected_mass = pd.DataFrame()

    # for major, minors in [
    #     ("FeO", ["MnO", "NiO", "CoO"]),
    #     ("CaO", ["SrO", "BaO"]),
    #     ("K2O", ["Rb2O", "Cs2O"]),
    #     ("Na2O", ["Li2O"]),
    #     ("Cr2O3", ["V2O3"]),
    # ]:
    #     _aggregate_components(df, major, minors, corrected_mass)

    # # Corrected molecular weight of Ca, Na and Fe
    # corrected_mass["Ca"] = corrected_mass["CaO"] - pt.O.mass
    # corrected_mass["Na"] = (corrected_mass["Na2O"] - pt.O.mass) / 2
    # corrected_mass["Fe"] = corrected_mass["FeO"] - pt.O.mass

    # # Get mineral data, update with corrected masses

    # minerals = {
    #     k: {**v} for k, v in NORM_MINERALS.items()
    # }  # copy the dictionary rather than edit it

    # _update_molecular_masses(minerals, corrected_mass)

    # Minor oxide combinations

    df["n_FeO_corr"] = df["FeO"] + df["MnO"] + df["NiO"] + df["CoO"]
    df["n_CaO_corr"] = df["CaO"] + df["SrO"] + df["BaO"]
    df["n_K2O_corr"] = df["K2O"] + df["Rb2O"] + df["Cs2O"]
    df["n_Na2O_corr"] = df["Na2O"] + df["Li2O"]
    df["n_Cr2O3_corr"] = df["Cr2O3"] + df["V2O3"]

    # Corrected oxide molecular weight computations
    df["x_MnO"] = df["MnO"] / df["n_FeO_corr"]
    df["x_FeO"] = df["FeO"] / df["n_FeO_corr"]

    df["x_NiO"] = df["NiO"] / df["n_FeO_corr"]
    df["x_CoO"] = df["CoO"] / df["n_FeO_corr"]

    df["x_SrO"] = df["SrO"] / df["n_CaO_corr"]
    df["x_BaO"] = df["BaO"] / df["n_CaO_corr"]
    df["x_CaO"] = df["CaO"] / df["n_CaO_corr"]

    df["x_Rb2O"] = df["Rb2O"] / df["n_K2O_corr"]
    df["x_Cs2O"] = df["Cs2O"] / df["n_K2O_corr"]
    df["x_K2O"] = df["K2O"] / df["n_K2O_corr"]

    df["x_Li2O"] = df["Li2O"] / df["n_Na2O_corr"]
    df["x_Na2O"] = df["Na2O"] / df["n_Na2O_corr"]

    df["x_V2O3"] = df["V2O3"] / df["n_Cr2O3_corr"]
    df["x_Cr2O3"] = df["Cr2O3"] / df["n_Cr2O3_corr"]

    df["FeO"] = df["n_FeO_corr"]
    df["CaO"] = df["n_CaO_corr"]
    df["K2O"] = df["n_K2O_corr"]
    df["Na2O"] = df["n_Na2O_corr"]
    df["Cr2O3"] = df["n_Cr2O3_corr"]

    # Corrected normative mineral molecular weight computations

    def corr_m_wt(oxide):
        return df["x_" + oxide] * pt.formula(oxide).mass

    df["MW_FeO_corr"] = (
        corr_m_wt("MnO") + corr_m_wt("NiO") + corr_m_wt("CoO") + corr_m_wt("FeO")
    )
    df["MW_CaO_corr"] = corr_m_wt("BaO") + corr_m_wt("SrO") + corr_m_wt("CaO")
    df["MW_K2O_corr"] = corr_m_wt("Rb2O") + corr_m_wt("Cs2O") + corr_m_wt("K2O")
    df["MW_Na2O_corr"] = corr_m_wt("Li2O") + corr_m_wt("Na2O")
    df["MW_Cr2O3_corr"] = corr_m_wt("V2O3") + corr_m_wt("Cr2O3")

    # Corrected molecular weight of Ca, Na and Fe
    df["MW_Ca_corr"] = df["MW_CaO_corr"] - pt.O.mass
    df["MW_Na_corr"] = (df["MW_Na2O_corr"] - pt.O.mass) / 2
    df["MW_Fe_corr"] = df["MW_FeO_corr"] - pt.O.mass

    for m in ["Q", "Z", "C", "Ru", "Hm", "Mg-Ol", "Mg-Hy"]:
        minerals[m]["mass"] = pt.formula(minerals[m]["formulae"]).mass

    minerals["Fe-Hy"]["mass"] = df["MW_FeO_corr"] + 60.0843
    minerals["Fe-Ol"]["mass"] = (2 * df["MW_FeO_corr"]) + 60.0843
    minerals["Mt"]["mass"] = df["MW_FeO_corr"] + 159.6882
    minerals["Il"]["mass"] = df["MW_FeO_corr"] + 79.8658
    minerals["An"]["mass"] = df["MW_CaO_corr"] + 222.129876
    minerals["Mg-Di"]["mass"] = df["MW_CaO_corr"] + 160.4730
    minerals["Wo"]["mass"] = df["MW_CaO_corr"] + 60.0843
    minerals["Cs"]["mass"] = 2 * df["MW_CaO_corr"] + 60.0843
    minerals["Tn"]["mass"] = df["MW_CaO_corr"] + 139.9501
    minerals["Pf"]["mass"] = df["MW_CaO_corr"] + 79.8558
    minerals["CaF2-Ap"]["mass"] = (
        3 * df["MW_CaO_corr"] + (1 / 3) * df["MW_Ca_corr"] + 154.6101241
    )
    minerals["Ap"]["mass"] = (10 / 3) * df["MW_CaO_corr"] + 141.944522
    minerals["Cc"]["mass"] = df["MW_CaO_corr"] + 44.0095
    minerals["Ab"]["mass"] = df["MW_Na2O_corr"] + 462.467076
    minerals["Ne"]["mass"] = df["MW_Na2O_corr"] + 222.129876
    minerals["Th"]["mass"] = df["MW_Na2O_corr"] + 80.0642
    minerals["Nc"]["mass"] = df["MW_Na2O_corr"] + 44.0095
    minerals["Ac"]["mass"] = df["MW_Na2O_corr"] + 400.0254
    minerals["Ns"]["mass"] = df["MW_Na2O_corr"] + 60.0843
    minerals["Or"]["mass"] = df["MW_K2O_corr"] + 462.467076
    minerals["Lc"]["mass"] = df["MW_K2O_corr"] + 342.298476
    minerals["Kp"]["mass"] = df["MW_K2O_corr"] + 222.129876
    minerals["Ks"]["mass"] = df["MW_K2O_corr"] + 60.0843
    minerals["Fe-Di"]["mass"] = df["MW_FeO_corr"] + df["MW_CaO_corr"] + 120.1686
    minerals["Cm"]["mass"] = df["MW_FeO_corr"] + df["MW_Cr2O3_corr"]
    minerals["Hl"]["mass"] = df["MW_Na_corr"] + 35.4527
    minerals["Fr"]["mass"] = df["MW_Ca_corr"] + 37.9968064
    minerals["Pr"]["mass"] = df["MW_Fe_corr"] + 64.132

    df["Y"] = 0

    ############################################################################
    # Calculate normative components
    ############################################################################

    # Normative Zircon
    df["Z"] = df["ZrO2"]
    df["Y"] = df["Z"]

    # Normative apatite

    df["Ap"] = np.where(
        df["CaO"] >= (3 + 1 / 3) * df["P2O5"], df["P2O5"], df["CaO"] / (3 + 1 / 3)
    ).T

    df["CaO_"] = np.where(
        df["CaO"] >= (3 + 1 / 3) * df["P2O5"], df["CaO"] - (3 + 1 / 3) * df["Ap"], 0
    ).T

    df["P2O5_"] = np.where(
        df["CaO"] < (3 + 1 / 3) * df["P2O5"], df["P2O5"] - df["Ap"], 0
    ).T

    df["CaO"] = df["CaO_"]
    df["P2O5"] = df["P2O5_"]

    df["FREE_P2O5"] = df["P2O5"]

    # apatite options where F in present

    df["ap_option"] = np.where(df["F"] >= (2 / 3) * df["Ap"], 2, 3).T

    df["F"] = np.where(
        (df["ap_option"]) == 2 & (df["F"] > 0), df["F"] - (2 / 3 * df["Ap"]), df["F"]
    ).T

    df["CaF2-Ap"] = np.where((df["ap_option"]) == 3 & (df["F"] > 0), df["F"] * 1.5, 0).T

    df["CaO-Ap"] = np.where(
        (df["ap_option"]) == 3 & (df["F"] > 0), df["P2O5"] - (1.5 * df["F"]), 0
    ).T

    df["Ap"] = np.where(
        (df["ap_option"]) == 3 & (df["F"] > 0), df["CaF2-Ap"] + df["CaO-Ap"], df["Ap"]
    ).T

    df["FREEO_12b"] = np.where(df["ap_option"] == 2, 1 / 3 * df["Ap"], 0).T
    df["FREEO_12c"] = np.where(df["ap_option"] == 3, df["F"] / 2, 0).T

    # Normative Fluorite
    df["Fr"] = np.where(df["CaO"] >= df["F"] / 2, df["F"] / 2, df["CaO"]).T

    df["CaO"] = np.where(df["CaO"] >= df["F"] / 2, df["CaO"] - df["Fr"], 0).T

    df["F"] = np.where(df["CaO"] >= df["F"] / 2, df["F"], df["F"] - (2 * df["Fr"])).T

    df["FREEO_13"] = df["Fr"]
    df["FREE_F"] = df["F"]

    # Normative halite
    df["Hl"] = np.where(df["Na2O"] >= 2 * df["Cl"], df["Cl"], df["Na2O"] / 2).T

    df["Na2O"] = np.where(df["Na2O"] >= 2 * df["Cl"], df["Na2O"] - df["Hl"] / 2, 0).T

    df["Cl"] = np.where(df["Na2O"] >= 2 * df["Cl"], df["Cl"], df["Cl"] - df["Hl"]).T

    df["FREE_Cl"] = df["Cl"]
    df["FREEO_14"] = df["Hl"] / 2

    # Normative thenardite
    df["Th"] = np.where(df["Na2O"] >= df["SO3"], df["SO3"], df["Na2O"]).T

    df["Na2O"] = np.where(df["Na2O"] >= df["SO3"], df["Na2O"] - df["Th"], 0).T

    df["SO3"] = np.where(df["Na2O"] >= df["SO3"], df["SO3"], df["SO3"] - df["Th"]).T

    df["FREE_SO3"] = df["SO3"]

    # Normative Pyrite
    df["Pr"] = np.where(df["FeO"] >= 2 * df["S"], df["S"] / 2, df["FeO"]).T

    df["FeO"] = np.where(
        df["FeO"] >= 2 * df["S"], df["FeO"] - df["Pr"], df["FeO"] - df["Pr"] * 2
    ).T

    df["FREE_S"] = np.where(df["FeO"] >= 2 * df["S"], 0, df["FeO"]).T

    df["FeO"] = df["FeO"] - df["FREE_S"]

    df["FREEO_16"] = df["Pr"]

    # Normative sodium carbonate or calcite

    df["Nc"] = np.where(df["Na2O"] >= df["CO2"], df["CO2"], df["Na2O"]).T

    df["Na2O"] = np.where(df["Na2O"] >= df["CO2"], df["Na2O"] - df["Nc"], df["Na2O"]).T

    df["CO2"] = np.where(df["Na2O"] >= df["CO2"], df["CO2"], df["CO2"] - df["Nc"]).T

    df["Cc"] = np.where(df["CaO"] >= df["CO2"], df["CO2"], df["CaO"]).T

    df["CaO"] = np.where(df["Na2O"] >= df["CO2"], df["CaO"] - df["Cc"], df["CaO"]).T

    df["CO2"] = np.where(df["Na2O"] >= df["CO2"], df["CO2"], df["CO2"] - df["Cc"]).T

    df["FREECO2"] = df["CO2"]

    # Normative Chromite
    df["Cm"] = np.where(df["FeO"] >= df["Cr2O3"], df["Cr2O3"], df["FeO"]).T

    df["FeO"] = np.where(df["FeO"] >= df["Cr2O3"], df["FeO"] - df["Cm"], 0).T
    df["Cr2O3"] = np.where(
        df["FeO"] >= df["Cr2O3"], df["Cr2O3"] - df["Cm"], df["Cr2O3"]
    ).T

    df["FREE_CR2O3"] = df["Cm"]

    # Normative Ilmenite
    df["Il"] = np.where(df["FeO"] >= df["TiO2"], df["TiO2"], df["FeO"]).T

    df["FeO_"] = np.where(df["FeO"] >= df["TiO2"], df["FeO"] - df["Il"], 0).T

    df["TiO2_"] = np.where(df["FeO"] >= df["TiO2"], 0, df["TiO2"] - df["Il"]).T

    df["FeO"] = df["FeO_"]

    df["TiO2"] = df["TiO2_"]

    # Normative Orthoclase/potasium metasilicate

    df["Or_p"] = np.where(df["Al2O3"] >= df["K2O"], df["K2O"], df["Al2O3"]).T

    df["Al2O3_"] = np.where(df["Al2O3"] >= df["K2O"], df["Al2O3"] - df["Or_p"], 0).T

    df["K2O_"] = np.where(df["Al2O3"] >= df["K2O"], 0, df["K2O"] - df["Or_p"]).T

    df["Ks"] = df["K2O_"]

    df["Y"] = np.where(
        df["Al2O3"] >= df["K2O"],
        df["Y"] + (df["Or_p"] * 6),
        df["Y"] + (df["Or_p"] * 6 + df["Ks"]),
    ).T

    df["Al2O3"] = df["Al2O3_"]
    df["K2O"] = df["K2O_"]

    # Normative Albite
    df["Ab_p"] = np.where(df["Al2O3"] >= df["Na2O"], df["Na2O"], df["Al2O3"]).T

    df["Al2O3_"] = np.where(df["Al2O3"] >= df["Na2O"], df["Al2O3"] - df["Ab_p"], 0).T

    df["Na2O_"] = np.where(df["Al2O3"] >= df["Na2O"], 0, df["Na2O"] - df["Ab_p"]).T

    df["Y"] = df["Y"] + (df["Ab_p"] * 6)

    df["Al2O3"] = df["Al2O3_"]
    df["Na2O"] = df["Na2O_"]

    # Normative Acmite / sodium metasilicate
    df["Ac"] = np.where(df["Na2O"] >= df["Fe2O3"], df["Fe2O3"], df["Na2O"]).T

    df["Na2O_"] = np.where(df["Na2O"] >= df["Fe2O3"], df["Na2O"] - df["Ac"], 0).T

    df["Fe2O3_"] = np.where(df["Na2O"] >= df["Fe2O3"], 0, df["Fe2O3"] - df["Ac"]).T

    df["Ns"] = df["Na2O_"]

    df["Y"] = np.where(
        df["Na2O"] >= df["Fe2O3"],
        df["Y"] + (4 * df["Ac"] + df["Ns"]),
        df["Y"] + 4 * df["Ac"],
    ).T

    df["Na2O"] = df["Na2O_"]
    df["Fe2O3"] = df["Fe2O3_"]

    # Normative Anorthite / Corundum
    df["An"] = np.where(df["Al2O3"] >= df["CaO"], df["CaO"], df["Al2O3"]).T

    df["Al2O3_"] = np.where(df["Al2O3"] >= df["CaO"], df["Al2O3"] - df["An"], 0).T

    df["CaO_"] = np.where(df["Al2O3"] >= df["CaO"], 0, df["CaO"] - df["An"]).T

    df["C"] = df["Al2O3_"]

    df["Al2O3"] = df["Al2O3_"]

    df["CaO"] = df["CaO_"]

    df["Y"] = df["Y"] + 2 * df["An"]

    # Normative Sphene / Rutile
    df["Tn_p"] = np.where(df["CaO"] >= df["TiO2"], df["TiO2"], df["CaO"]).T

    df["CaO_"] = np.where(df["CaO"] >= df["TiO2"], df["CaO"] - df["Tn_p"], 0).T

    df["TiO2_"] = np.where(df["CaO"] >= df["TiO2"], 0, df["TiO2"] - df["Tn_p"]).T

    df["CaO"] = df["CaO_"]
    df["TiO2"] = df["TiO2_"]

    df["Ru"] = df["TiO2"]

    df["Y"] = df["Y"] + df["Tn_p"]

    # Normative Magnetite / Hematite
    df["Mt"] = np.where(df["Fe2O3"] >= df["FeO"], df["FeO"], df["Fe2O3"]).T

    df["Fe2O3_"] = np.where(df["Fe2O3"] >= df["FeO"], df["Fe2O3"] - df["Mt"], 0).T

    df["FeO_"] = np.where(df["Fe2O3"] >= df["FeO"], 0, df["FeO"] - df["Mt"]).T

    df["Fe2O3"] = df["Fe2O3_"]

    df["FeO"] = df["FeO_"]

    df["Hm"] = df["Fe2O3"]

    # Subdivision of some normative minerals
    df["MgFe_O"] = df[["FeO", "MgO"]].sum(axis=1)

    df["MgO_ratio"] = df["MgO"] / df["MgFe_O"]
    df["FeO_ratio"] = df["FeO"] / df["MgFe_O"]

    # Provisional normative diopside, wollastonite / Hypersthene
    df["Di_p"] = np.where(df["CaO"] >= df["MgFe_O"], df["MgFe_O"], df["CaO"]).T

    df["CaO_"] = np.where(df["CaO"] >= df["MgFe_O"], df["CaO"] - df["Di_p"], 0).T

    df["MgFe_O_"] = np.where(df["CaO"] >= df["MgFe_O"], 0, df["MgFe_O"] - df["Di_p"]).T

    df["Hy_p"] = df["MgFe_O_"]

    df["Wo_p"] = np.where(df["CaO"] >= df["MgFe_O"], df["CaO_"], 0).T

    df["Y"] = np.where(
        df["CaO"] >= df["MgFe_O"],
        df["Y"] + (2 * df["Di_p"] + df["Wo_p"]),
        df["Y"] + (2 * df["Di_p"] + df["Hy_p"]),
    ).T

    df["CaO"] = df["CaO_"]
    df["MgFe_O"] = df["MgFe_O_"]

    # Normative quartz / undersaturated minerals
    df["Q"] = np.where(df["SiO2"] >= df["Y"], df["SiO2"] - df["Y"], 0).T

    df["D"] = np.where(df["SiO2"] < df["Y"], df["Y"] - df["SiO2"], 0).T

    df["deficit"] = df["D"] > 0

    # Normative Olivine / Hypersthene
    df["Ol_"] = np.where((df["D"] < df["Hy_p"] / 2), df["D"], df["Hy_p"] / 2).T

    df["Hy"] = np.where((df["D"] < df["Hy_p"] / 2), df["Hy_p"] - 2 * df["D"], 0).T

    df["D1"] = df["D"] - df["Hy_p"] / 2

    df["Ol"] = np.where((df["deficit"]), df["Ol_"], 0).T

    df["Hy"] = np.where((df["deficit"]), df["Hy"], df["Hy_p"]).T

    df["deficit"] = df["D1"] > 0

    # Normative Sphene / Perovskite
    df["Tn"] = np.where((df["D1"] < df["Tn_p"]), df["Tn_p"] - df["D1"], 0).T

    df["Pf_"] = np.where((df["D1"] < df["Tn_p"]), df["D1"], df["Tn_p"]).T

    df["D2"] = df["D1"] - df["Tn_p"]

    df["Tn"] = np.where((df["deficit"]), df["Tn"], df["Tn_p"]).T
    df["Pf"] = np.where((df["deficit"]), df["Pf_"], 0).T

    df["deficit"] = df["D2"] > 0

    # Normative Nepheline / Albite
    df["Ne_"] = np.where((df["D2"] < 4 * df["Ab_p"]), df["D2"] / 4, df["Ab_p"]).T

    df["Ab"] = np.where((df["D2"] < 4 * df["Ab_p"]), df["Ab_p"] - df["D2"] / 4, 0).T

    df["D3"] = df["D2"] - 4 * df["Ab_p"]

    df["Ne"] = np.where((df["deficit"]), df["Ne_"], 0).T
    df["Ab"] = np.where((df["deficit"]), df["Ab"], df["Ab_p"]).T

    df["deficit"] = df["D3"] > 0

    # Normative Leucite / Orthoclase
    df["Lc"] = np.where((df["D3"] < 2 * df["Or_p"]), df["D3"] / 2, df["Or_p"]).T

    df["Or"] = np.where((df["D3"] < 2 * df["Or_p"]), df["Or_p"] - df["D3"] / 2, 0).T

    df["D4"] = df["D3"] - 2 * df["Or_p"]

    df["Lc"] = np.where((df["deficit"]), df["Lc"], 0).T
    df["Or"] = np.where((df["deficit"]), df["Or"], df["Or_p"]).T

    df["deficit"] = df["D4"] > 0

    # Normative dicalcium silicate / wollastonite
    df["Cs"] = np.where((df["D4"] < df["Wo_p"] / 2), df["D4"], df["Wo_p"] / 2).T

    df["Wo"] = np.where((df["D4"] < df["Wo_p"] / 2), df["Wo_p"] - 2 * df["D4"], 0).T

    df["D5"] = df["D4"] - df["Wo_p"] / 2

    df["Cs"] = np.where((df["deficit"]), df["Cs"], 0).T
    df["Wo"] = np.where((df["deficit"]), df["Wo"], df["Wo_p"]).T

    df["deficit"] = df["D5"] > 0

    # Normative dicalcium silicate / Olivine Adjustment
    df["Cs_"] = np.where(
        (df["D5"] < df["Di_p"]), df["D5"] / 2 + df["Cs"], df["Di_p"] / 2 + df["Cs"]
    ).T

    df["Ol_"] = np.where(
        (df["D5"] < df["Di_p"]), df["D5"] / 2 + df["Ol"], df["Di_p"] / 2 + df["Ol"]
    ).T

    df["Di_"] = np.where((df["D5"] < df["Di_p"]), df["Di_p"] - df["D5"], 0).T

    df["D6"] = df["D5"] - df["Di_p"]

    df["Cs"] = np.where((df["deficit"]), df["Cs_"], df["Cs"]).T
    df["Ol"] = np.where((df["deficit"]), df["Ol_"], df["Ol"]).T
    df["Di"] = np.where((df["deficit"]), df["Di_"], df["Di_p"]).T

    df["deficit"] = df["D6"] > 0

    # Normative Kaliophilite / Leucite
    df["Kp"] = np.where((df["Lc"] >= df["D6"] / 2), df["D6"] / 2, df["Lc"]).T

    df["Lc_"] = np.where((df["Lc"] >= df["D6"] / 2), df["Lc"] - df["D6"] / 2, 0).T

    df["Kp"] = np.where((df["deficit"]), df["Kp"], 0).T
    df["Lc"] = np.where((df["deficit"]), df["Lc_"], df["Lc"]).T

    df["DEFSIO2"] = np.where(
        (df["Lc"] < df["D6"] / 2) & (df["deficit"]), df["D6"] - 2 * df["Kp"], 0
    ).T
    ############################################################################
    # Allocate definite mineral proportions
    # Subdivide Hypersthene, Diopside and Olivine into Mg- and Fe- varieties
    # TODO: Add option for subdivision?

    df["Fe-Hy"] = df["Hy"] * df["FeO_ratio"]
    df["Fe-Di"] = df["Di"] * df["FeO_ratio"]
    df["Fe-Ol"] = df["Ol"] * df["FeO_ratio"]

    df["Mg-Hy"] = df["Hy"] * df["MgO_ratio"]
    df["Mg-Di"] = df["Di"] * df["MgO_ratio"]
    df["Mg-Ol"] = df["Ol"] * df["MgO_ratio"]

    ############################################################################
    # calculate free component molecular abundances
    ############################################################################
    FREE = pd.DataFrame()

    FREE["FREEO_12b"] = (
        (1 + ((0.1) * ((minerals["CaF2-Ap"]["mass"] / 328.86918) - 1)))
        * pt.O.mass
        * df["FREEO_12b"]
    )
    FREE["FREEO_12c"] = (
        (
            1
            + (
                (0.1)
                * (df["CaF2-Ap"] / df["Ap"])
                * ((minerals["CaF2-Ap"]["mass"] / 328.86918) - 1)
            )
        )
        * pt.O.mass
        * df["FREEO_12c"]
    )

    FREE["FREEO_13"] = (
        (1 + ((pt.formula("CaO").mass / 56.0774) - 1)) * pt.O.mass * df["FREEO_13"]
    )

    FREE["FREEO_14"] = (
        (1 + (0.5 * ((pt.formula("Na2O").mass / 61.9789) - 1)))
        * pt.O.mass
        * df["FREEO_14"]
    )

    FREE["FREEO_16"] = (
        (1 + ((pt.formula("FeO").mass / 71.8444) - 1)) * pt.O.mass * df["FREEO_16"]
    )

    FREE["O"] = FREE[
        ["FREEO_12b", "FREEO_12c", "FREEO_13", "FREEO_14", "FREEO_16"]
    ].sum(axis=1)

    ############################################################################
    # get masses of free components
    ############################################################################
    FREE["CO2"] = df["FREECO2"] * pt.formula("CO2").mass  # 44.0095

    FREE["P2O5"] = df["FREE_P2O5"] * pt.formula("P2O5").mass  # 141.94452
    FREE["F"] = df["FREE_F"] * pt.F.mass  # 18.9984032
    FREE["Cl"] = df["FREE_Cl"] * pt.Cl.mass  # 35.4527
    FREE["SO3"] = df["FREE_SO3"] * pt.formula("SO3").mass  # 80.0642
    FREE["S"] = df["FREE_S"] * pt.S.mass  # 32.066
    FREE["Cr2O3"] = df["FREE_CR2O3"] * pt.formula("Cr2O3").mass  # 151.990

    FREE["OXIDES"] = FREE[["P2O5", "F", "Cl", "SO3", "S", "Cr2O3"]].sum(axis=1)

    FREE["DEFSIO2"] = df["DEFSIO2"] * pt.formula("SiO2").mass  # 60.0843
    FREE.drop(["P2O5", "F", "Cl", "SO3", "S", "Cr2O3"], axis=1, inplace=True)

    ############################################################################
    # populate tables of molecular proportions and masses
    ############################################################################
    mineral_proportions = pd.DataFrame()
    mineral_pct_mm = pd.DataFrame()

    for mineral in minerals.keys():
        if mineral == ["Ap"]:
            # deal with the results of apatite options
            # get the abundance weighted total mass of apatite where split
            # otherwise just get the apatite mass
            mineral_pct_mm[mineral] = np.where(
                df["ap_option"] == 2,
                df[mineral] * minerals["Ap"]["mass"],
                (df["CaF2-Ap"] * minerals["CaF2-Ap"]["mass"])
                + (df["CaO-Ap"] * minerals["Ap"]["mass"]),
            )
        else:
            mineral_proportions[mineral] = df[mineral]
            mineral_pct_mm[mineral] = (
                mineral_proportions[mineral] * minerals[mineral]["mass"]
            )

    # rename columns with proper names rather than abbreviations
    mineral_pct_mm.columns = [
        minerals[mineral]["name"] for mineral in mineral_pct_mm.columns
    ]
    mineral_pct_mm.fillna(0, inplace=True)

    return mineral_pct_mm
