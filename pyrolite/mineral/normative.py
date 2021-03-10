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


def CIPW_norm(df, form="weight"):
    """
    Standardised calcuation of estimated mineralogy from bulk rock chemistry.
    Takes a dataframe of chemistry and creates a dataframe of estimated mineralogy.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe containing compositions to transform.

    Returns
    --------
    :class:`pandas.DataFrame`

    Notes
    -----
    This function is currently a stub.
    """

    # devolatilise and renormalise

    # deal with Fe oxidation

    # convert to molar data

    # aggregate like-elements to their most commom member
    # Ferric : FeO, MnO, NiO, CoO
    # Calcic : CaO, BaO, SrO
    # Potassic : K2O, Rb2O, Cs2O
    # Sodic : Na2O, Li2O
    # Chromic : Cr2O3, V2O3

    # recalculate 'molecular weigths' for each of these components, and the minerals

    # Zircon

    # Apatite

    # Flourite

    # halite

    # thenardite

    # pyrite

    # cancrinite or calcite

    # chromite

    # ilmenite

    # orthoclase / potassium metasilicate

    # albite

    # acmite / sodium metasilicate

    # anorthite / corundum

    # titanite / rutile

    # magnetite / hematite

    # diopside / wollastonite / hypersthene

    # quartz / undersadurated minerals

    # silica deficiency?

    # olivine / hypersthene

    # titanite / perobskite

    # nephenline / albite

    # leucite / orthoclase

    # dicalcium silicate / wollastonite

    # dicalcium silicate / olivine adjustment

    # kaliophilite / leucite

    # allocate provisional mineral proportions to definite ones

    # normalise

    # calculate Salic, Femic, C.I., D.I

    pass
