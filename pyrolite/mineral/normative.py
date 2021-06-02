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


def CIPW_norm(df):
    """
    Standardised calcuation of estimated mineralogy from bulk rock chemistry.
    Takes a dataframe of chemistry & creates a dataframe of estimated mineralogy.

    This is the CIPW norm of Verma et al. (2003).  This version only uses major
    elements.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe containing compositions to transform.

    Returns
    --------
    :class:`pandas.DataFrame`

    Todo
    ----
    * Note whether data needs to be normalised to 1 or 100?
    """

    columns = ['SiO2','TiO2','Al2O3','Fe2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5','CO2','SO3']
    critical_columns = ['SiO2','TiO2','Al2O3','Fe2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5']
    # Validate columns
    if np.all(data.columns == columns): # pass if all columns present
        pass
    else: # raise warning for missing columns
        for col in critical_columns:
            if col not in data.columns:
                logger.warning('%s columns missing', col)
    
    # Check that all of the columns we'd like are present
    to_impute = []
    if not set(columns).issubset(set(df.columns.values)):
        # raise warning for missing columns
        missing_critical = [c for c in critical_columns if c not in df.columns]
        if missing_critical:
            logger.warning(
                "Required columns missing: {}".format(", ".join(missing_critical))
            )
        to_impute += [c for c in columns if c not in df.columns]

    # Reindex columns to be expected and fill missing ones with zeros
    df_update = df.reindex(columns=columns).fillna(0)
    if to_impute:  # Note that we're adding the columns with default values.
        logging.debug("Columns added with default 0 values: ", ", ".join(to_impute))

    df = df_update

    # Check if Fe2O3 and FeO are present
    if "FeO" and "Fe2O3" in df.columns:
        pass
    else:
        # should be able to do this in a vectorised way...
        for index, row in df.iterrows():
            Fe_ratio = (
                0.93 - (0.0042 * row["SiO2"]) - 0.022 * (row["Na2O"] + row["K2O"])
            )  # Middlemost
            if row["Fe2O3"] == 0:  # if Fe given as FeO
                df.loc[index, "FeO"] = row["FeO"] * Fe_ratio
                df.loc[index, "Fe2O3"] = (1.11134 * row["FeO"]) * (1 - Fe_ratio)
            if row["FeO"] == 0:  # if Fe given as Fe2O3
                df.loc[index, "FeO"] = (row["Fe2O3"] / 1.11134) * Fe_ratio
                df.loc[index, "Fe2O3"] = row["Fe2O3"] * (1 - Fe_ratio)

    # Normalise to 100 on anhydrous basis
    # res = df[columns].pyrochem.to_molecular()
    res = df.divide(df.sum(axis=1) / 100, axis=0)
    res = df.div(pd.Series([pt.formula(c).mass for c in columns], index=columns))

    # endmember component fractions
    res.pyrochem.add_MgNo(molecular=True)
    MgNo, FeNo = res["Mg#"], 1 - res["Mg#"]
    xFeO = res["FeO"] / (res["FeO"] + res["MnO"])
    xMnO = 1 - xFeO

    ############################################################################
    # Calculate effective molecular weights for silicate & oxide endmembers
    ############################################################################
    # When updating for trace elements will need to add more here
    mineral_mw = {}

    # Components (used only for calc) ##########################################
    # effective MW of (Fe2+, Mn2+)O
    molw_FeO = (xMnO * pt.formula("MnO") + xFeO * pt.formula("FeO")).mass
    # effective MW of Mn-fayalite
    molw_fayalite = (2 * molw_FeO) + pt.formula("SiO2").mass
    # effective v of ferrosilite (Fe2+, Mn2+)SiO3
    molw_fs = molw_FeO + pt.formula("SiO2").mass
    # Endmembers (used in the norm) ############################################
    # effective MW of ilmenite (Fe2+, Mn2+)TiO3
    molw_il = molw_FeO + pt.formula("TiO2").mass
    # effective MW of hypersthene (Mg2+, Fe2+, Mn2+)SiO3
    molw_hyp = pt.formula("MgSiO3").mass * MgNo + molw_fs * FeNo
    # effective MW of olivine (Mg2+, Fe2+, Mn2+)SiO4
    molw_ol = pt.formula("Mg2SiO4").mass * MgNo + molw_fayalite * FeNo
    # effective MW of magnetite
    molw_mt = molw_FeO + pt.formula("Fe2O3").mass

    for n, mw in zip(
        ["ilmenite", "hypersthene", "olivine", "magnetite"],
        [molw_il, molw_hyp, molw_ol, molw_mt],
    ):
        mineral_mw[n] = mw
    # anhydrous apatite (Ca10P6O25)/3 relative to P2O5
    mineral_mw["apatite"] = (
        pt.formula("Ca10(PO4)6(OH)2").mass - pt.formula("H2O").mass
    ) / 3
    mineral_mw["halite"] = pt.formula("NaCl").mass  # leave out Cl?
    mineral_mw["fluorite"] = pt.formula("CaF2").mass  # leave out F2?
    mineral_mw["pyrite"] = pt.formula("FeS2").mass
    ############################################################################
    # Aggregate Mn, Fe
    ############################################################################
    res["FeO"] = res["FeO"] + res["MnO"]
    res = res.drop(["MnO"], axis=1)

    ############################################################################
    # Calculate normative components
    ############################################################################
    # Below two dataframes are used to store i) a reservoir composition from
    # which minerals are 'extracted' and ii) the mineralogical norm extracted
    # from it. The silica content of the norm is denoted 'Y_SiO2', and the
    # deficit in silica relative to the reservoir is denonted 'D_SiO2'.

    norm = pd.DataFrame(index=res.index)  # dataframe to store mineralogy
    # Apatite ###################
    ap_fltr = res["CaO"] >= ((10.0 / 3) * res["P2O5"])  # where phosphate is limiting
    norm["apatite"] = np.where(ap_fltr, res["P2O5"], res["CaO"] / (10.0 / 3))
    res["CaO"] = res["CaO"] - (10.0 / 3) * norm["apatite"]
    res["P2O5"] = res["P2O5"] - norm["apatite"]
    # Fluorite
    # Halite
    # Thenardite ###################
    then_fltr = res["Na2O"] >= res["SO3"]  # where sulfate is limiting
    norm["thenardite"] = np.where(then_fltr, res["SO3"], res["Na2O"])
    res["SO3"] = res["SO3"] - norm["thenardite"]
    res["Na2O"] = res["Na2O"] - norm["thenardite"]

    # Ilmenite ###################
    ilm_fltr = res["FeO"] >= res["TiO2"]  # where titanium is limiting
    norm["ilmenite"] = np.where(ilm_fltr, res["TiO2"], res["FeO"])
    res["FeO"] = res["FeO"] - norm["ilmenite"]
    res["TiO2"] = res["TiO2"] - norm["ilmenite"]

    # Orthoclase ###################
    ort_fltr = res["Al2O3"] >= res["K2O"]  # where aluminium is limiting
    norm["orthoclase"] = np.where(ort_fltr, res["K2O"], res["Al2O3"])
    res["Al2O3"] = res["Al2O3"] - norm["orthoclase"]
    res["K2O"] = res["K2O"] - norm["orthoclase"]
    res["ks"] = res["K2O"]

    # Norm Silica Content
    Y_SiO2 = (norm["orthoclase"] * 6) + res["ks"]

    # Albite ###################
    alb_fltr = res["Al2O3"] >= res["Na2O"]  # where sodium is limiting
    norm["albite"] = np.where(alb_fltr, res["Na2O"], res["Al2O3"])
    res["Al2O3"] = res["Al2O3"] - norm["albite"]
    res["Na2O"] = res["Na2O"] - norm["albite"]

    Y_SiO2 = Y_SiO2 + (6 * norm["albite"])  # add to sum silica

    # Acmite - for Fe2O3 ###################
    acm_fltr = res["Na2O"] >= res["Fe2O3"]  # where iron is limiting
    norm["acmite"] = np.where(acm_fltr, res["Fe2O3"], res["Na2O"])
    res["Na2O"] = res["Na2O"] - norm["acmite"]
    norm["ns"] = res["Na2O"]  # leftover sodium assigned to sodium metasilicate
    res["Fe2O3"] = res["Fe2O3"] - norm["acmite"]

    Y_SiO2 = Y_SiO2 + (4 * norm["acmite"])

    # Anorthite and corundum ###################
    ano_fltr = res["Al2O3"] >= res["CaO"]  # where calcium is limiting
    norm["anorthite"] = np.where(ano_fltr, res["CaO"], res["Al2O3"])
    res["Al2O3"] = res["Al2O3"] - norm["anorthite"]
    norm["corundum"] = res["Al2O3"]  # assign any residual aluminium to corundum
    res["CaO"] = res["CaO"] - norm["anorthite"]

    Y_SiO2 = (2 * norm["anorthite"]) + Y_SiO2

    # Titanite/rutile ###################  - not sure if working correctly
    tit_fltr = res["CaO"] >= res["TiO2"]  # where titanium is limiting
    norm["titanite"] = np.where(tit_fltr, res["TiO2"], res["CaO"])
    res["CaO"] = res["CaO"] - norm["titanite"]
    res["TiO2"] = res["TiO2"] - norm["titanite"]
    norm["rutile"] = res["TiO2"]  # assign any residual titanium to rutile

    Y_SiO2 = Y_SiO2 + norm["titanite"]

    # Magnetite and haematite ###################
    mag_fltr = res["Fe2O3"] >= res["FeO"]  # where FeO is limiting
    norm["magnetite"] = np.where(mag_fltr, res["FeO"], res["Fe2O3"])
    res["Fe2O3"] = res["Fe2O3"] - norm["magnetite"]
    res["FeO"] = res["FeO"] - norm["magnetite"]
    norm["haematite"] = res["Fe2O3"]  # assign any residual Fe2O3 to haeamatite

    # Subdivided normative minerals
    res["MgFeO"] = res["MgO"] + res["FeO"]
    #     data['Mg/Fe'] = data['MgO']/(data['FeO']+data['MgO'])
    #     data['Fe/Mg'] = data['FeO']/(data['FeO']+data['MgO'])

    # Diopside, Wollastonite, Hypersthene (provisional) ###################
    dio_flt = res["CaO"] >= res["MgFeO"]  # where Mg and Fe are limiting
    norm["diopside"] = np.where(dio_flt, res["MgFeO"], res["CaO"])
    res["CaO"] = res["CaO"] - norm["diopside"]
    res["MgFeO"] = res["MgFeO"] - norm["diopside"]
    norm["wollastonite"] = res["CaO"]  # assign residual calcium to wollastonite
    # assign residual magnesium and iron to hypersthene
    norm["hypersthene"] = res["MgFeO"]

    Y_SiO2 += 2 * norm["diopside"] + norm["wollastonite"] + norm["hypersthene"]

    # Quartz/undersaturated minerals ###################
    qua_flt = res["SiO2"] >= Y_SiO2  # if silica is in excess
    norm["quartz"] = np.where(qua_flt, res["SiO2"] - Y_SiO2, 0)

    # Silica Deficit
    D_SiO2 = np.where(qua_flt, 0, Y_SiO2 - res["SiO2"])

    # Olivine ###################
    # Using two filters as we want to skip if deficit = 0
    oli_flt_1 = (D_SiO2 > 0) & (D_SiO2 < norm["hypersthene"] / 2)  # deficit < hyp
    oli_flt_2 = (D_SiO2 > 0) & (D_SiO2 >= norm["hypersthene"] / 2)  # deficit > hyp
    norm["olivine"] = np.where(oli_flt_1, D_SiO2, 0)
    norm["olivine"] = np.where(oli_flt_2, norm["hypersthene"] / 2, norm["olivine"])
    hypersthene = np.where(
        oli_flt_1, norm["hypersthene"] - (2 * D_SiO2), norm["hypersthene"]
    )
    hypersthene = np.where(oli_flt_2, 0, hypersthene)
    norm["hypersthene"] = hypersthene

    D_SiO2 = np.where(oli_flt_2, D_SiO2 - (norm["hypersthene"] / 2), 0)

    # Sphene/perovskite ###################
    # if silica deficit still present...
    tit_flt_1 = (D_SiO2 > 0) & (D_SiO2 < norm["titanite"])  # if  deficit < Ttn
    tit_flt_2 = (D_SiO2 > 0) & (D_SiO2 >= norm["titanite"])  # if  deficit >= Ttn
    titanite = np.where(tit_flt_1, norm["titanite"] - D_SiO2, norm["titanite"])
    titanite = np.where(tit_flt_2, 0, titanite)
    perovskite = np.where(tit_flt_1, D_SiO2, 0)
    norm["perovskite"] = np.where(tit_flt_2, norm["titanite"], perovskite)
    D_SiO2 = np.where(tit_flt_2, D_SiO2 - norm["titanite"], 0)
    norm["titanite"] = titanite
    norm["perovskite"] = perovskite  ## this is assigned twice? ##

    # Nepheline & Albite ###################
    nep_flt_1 = (D_SiO2 > 0) & (D_SiO2 < (4 * norm["albite"]))  # if deficit < Ab
    nep_flt_2 = (D_SiO2 > 0) & (D_SiO2 >= (4 * norm["albite"]))  # if deficit > Ab
    nepheline = np.where(nep_flt_1, D_SiO2 / 4, 0)
    nepheline = np.where(nep_flt_2, norm["albite"], nepheline)
    albite = np.where(nep_flt_1, norm["albite"] - (D_SiO2 / 4), norm["albite"])
    albite = np.where(nep_flt_2, 0, albite)

    norm["albite"] = albite
    norm["nepheline"] = nepheline

    D_SiO2 = np.where(nep_flt_2, D_SiO2 - 4 * (norm["albite"]), 0)

    # Leucite ###################
    leu_flt_1 = (D_SiO2 > 0) & (D_SiO2 < 2 * norm["orthoclase"])  # if deficit < Or
    leu_flt_2 = (D_SiO2 > 0) & (D_SiO2 >= 2 * norm["orthoclase"])  # if deficit > Or
    leucite = np.where(leu_flt_1, D_SiO2 / 2, 0)
    leucite = np.where(leu_flt_2, norm["orthoclase"], leucite)
    orthoclase = np.where(
        leu_flt_1, norm["orthoclase"] - (D_SiO2 / 2), norm["orthoclase"]
    )
    orthoclase = np.where(leu_flt_2, 0, norm["orthoclase"])
    norm["leucite"] = leucite
    norm["orthoclase"] = orthoclase

    D_SiO2 = np.where(leu_flt_2, D_SiO2 - (2 * norm["orthoclase"]), 0)

    # Dicalcium silicate/wollastonite ###################
    cs_flt_1 = (D_SiO2 > 0) & (D_SiO2 < norm["wollastonite"] / 2)  # if deficit < woll
    cs_flt_2 = (D_SiO2 > 0) & (D_SiO2 >= norm["wollastonite"] / 2)  # if deficit > woll
    cs = np.where(cs_flt_1, D_SiO2, 0)
    norm["cs"] = np.where(cs_flt_2, norm["wollastonite"], cs)
    wollastonite = np.where(
        cs_flt_1, norm["wollastonite"] - (2 * D_SiO2), norm["wollastonite"]
    )
    wollastonite = np.where(cs_flt_2, 0, wollastonite)
    norm["wollastonite"] = wollastonite

    D_SiO2 = np.where(cs_flt_2, D_SiO2 - (norm["wollastonite"] / 2), 0)

    # Dicalcium silicate/olivine ###################
    oli_flt_1 = (D_SiO2 > 0) & (D_SiO2 < norm["diopside"])  # if deficit < diopside
    oli_flt_2 = (D_SiO2 > 0) & (D_SiO2 >= norm["diopside"])  # if deficit > diopside
    olivine = np.where(oli_flt_1, norm["olivine"] + (D_SiO2 / 2), norm["olivine"])
    olivine = np.where(oli_flt_2, norm["olivine"] + (norm["diopside"] / 2), olivine)
    cs = np.where(oli_flt_1, norm["cs"] + (D_SiO2 / 2), norm["cs"])
    norm["cs"] = np.where(oli_flt_2, norm["cs"] + (norm["diopside"] / 2), cs)
    diopside = np.where(oli_flt_1, norm["diopside"] - D_SiO2, norm["diopside"])
    diopside = np.where(oli_flt_2, 0, diopside)
    D_SiO2 = np.where(oli_flt_2, D_SiO2 - norm["diopside"], 0)  # update silica deficit
    norm["diopside"] = diopside
    norm["olivine"] = olivine
    norm["cs"] = cs

    # Kaliophilite/leucite ###################
    kal_flt_1 = (D_SiO2 > 0) & (norm["leucite"] >= (D_SiO2 / 2))  # if deficit < leucite
    kal_flt_2 = (D_SiO2 > 0) & (norm["leucite"] < (D_SiO2 / 2))  # if deficit > leucite
    norm["kaliophilite"] = np.where(kal_flt_1, D_SiO2 / 2, 0)
    norm["kaliophilite"] = np.where(kal_flt_2, norm["leucite"], norm["kaliophilite"])
    leucite = np.where(kal_flt_1, norm["leucite"] - (D_SiO2 / 2), norm["leucite"])
    leucite = np.where(kal_flt_2, 0, leucite)
    norm["leucite"] = leucite
    # Subdivide hy & di into Mg & Fe??

    # final silica deficit
    D_SiO2 = np.where(kal_flt_2, D_SiO2 - (2 * norm["kaliophilite"]), 0)

    ############################################################################
    # Convert normative minerals to % by multiplying by molecular weights
    # Fe minerals using mw calculated earlier
    minerals = [
        ("apatite", mineral_mw["apatite"]),
        ("thenardite", pt.formula("Na2SO4").mass),
        ("ilmenite", mineral_mw["ilmenite"]),
        ("orthoclase", pt.formula("KAlSi3O8").mass * 2),
        ("albite", pt.formula("NaAlSi3O8").mass * 2),
        ("acmite", pt.formula("NaFe(SiO3)2").mass * 2),
        ("ns", pt.formula("Na2O SiO2").mass),
        ("anorthite", pt.formula("CaAl2Si2O8").mass),
        ("corundum", pt.formula("Al2O3").mass),
        ("titanite", pt.formula("CaTiSiO5").mass),
        ("rutile", pt.formula("TiO2").mass),
        ("magnetite", pt.formula("Fe3O4").mass),
        ("haematite", pt.formula("Fe2O3").mass),
        ("diopside", pt.formula("CaMgSi2O6").mass),
        ("wollastonite", pt.formula("CaSiO3").mass),
        ("hypersthene", mineral_mw["hypersthene"]),
        ("quartz", pt.formula("SiO2").mass),
        ("olivine", mineral_mw["olivine"]),
        ("perovskite", pt.formula("CaTiO3").mass),
        ("nepheline", pt.formula("NaAlSiO4").mass * 2),
        ("leucite", pt.formula("KAlSi2O6").mass * 2),
        ("cs", pt.formula("Ca2O 2SiO2").mass),
        ("kaliophilite", pt.formula("K2O Al2O3 2SiO2").mass),
    ]
    norm.loc[:, [m[0] for m in minerals]] *= np.array([m[1] for m in minerals])[None, :]
    return norm
