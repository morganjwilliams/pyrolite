import pandas as pd
import numpy as np
import periodictable as pt
from ..util.pd import to_frame
from ..util.log import Handle

logger = Handle(__name__)


def formula_to_elemental(formula, weight=True):
    """Convert a periodictable.formulas.Formula to elemental composition."""
    formula = pt.formula(formula)
    fmass = formula.mass
    composition = {}
    if weight:
        for a, c in formula.atoms.items():
            composition[str(a)] = (c * a.mass) / fmass
    else:
        atoms = sum([c for a, c in formula.atoms.items()])
        for a, c in formula.atoms.items():
            composition[str(a)] = c / atoms
    return composition


def merge_formulae(formulas):
    """
    Combine multiple formulae into one. Particularly useful for defining oxide mineral
    formulae.

    Parameters
    -----------
    formulas: iterable
        Iterable of multiple formulae to merge into a single larger molecular formulae.
    """
    molecule = pt.formula("")
    for f in formulas:
        molecule += pt.formula(f)
    return molecule


def recalc_cations(
    df,
    ideal_cations=4,
    ideal_oxygens=6,
    Fe_species=["FeO", "Fe", "Fe2O3"],
    oxygen_constrained=False,
):
    """
    Recalculate a composition to a.p.f.u.
    """
    assert ideal_cations is not None or ideal_oxygens is not None
    # if Fe2O3 and FeO are specified, calculate based on oxygen
    moles = to_frame(df)
    moles = moles.div([pt.formula(c).mass for c in moles.columns])
    moles = moles.where(~np.isclose(moles, 0.0), np.nan)

    # determine whether oxygen is an open or closed system
    Fe_species = [i for i in moles if i in Fe_species]  # keep dataframe ordering
    count_iron_species = len(Fe_species)
    oxygen_constrained = oxygen_constrained
    if not oxygen_constrained:
        if count_iron_species > 1:  # check that only one is defined
            oxygen_constrained = (
                count_iron_species
                - pd.isnull(moles.loc[:, Fe_species]).all(axis=1).sum()
            ) > 1

            if oxygen_constrained:
                logger.info("Multiple iron species defined. Calculating using oxygen.")
            else:
                logger.info("Single iron species defined. Calculating using cations.")

    components = moles.columns
    as_oxides = len(list(pt.formula(components[0]).atoms)) > 1
    schema = []
    # if oxygen_constrained:  # need to specifically separate Fe2 and Fe3
    if as_oxides:
        parts = [pt.formula(c).atoms for c in components]
        for p in parts:
            oxygens = p[pt.O]
            other_components = [i for i in list(p) if not i == pt.O]
            assert len(other_components) == 1  # need to be simple oxides
            other = other_components[0]
            charge = oxygens * 2 / p[other]
            ion = other.ion[charge]
            schema.append({str(ion): p[other], "O": oxygens})
    else:
        # elemental composition
        parts = components
        for part in parts:
            p = list(pt.formula(part).atoms)[0]
            if p.charge != 0:
                charge = p.charge
            else:
                charge = p.default_charge
            schema.append({p.ion[charge]: 1})

    ref = pd.DataFrame(data=schema)
    ref.columns = ref.columns.map(str)
    ref.index = components

    cation_masses = {c: pt.formula(c).mass for c in ref.columns}
    oxygen_index = [i for i in ref.columns if "O" in i][0]
    ref = ref.loc[:, [i for i in ref.columns if not i == oxygen_index] + [oxygen_index]]
    moles_ref = ref.copy(deep=True)
    moles_ref.loc[:, :] = (
        ref.values * moles.T.values
    )  # this works for series, not for frame

    moles_O = moles_ref[oxygen_index].sum()
    moles_cations = (
        moles_ref.loc[:, [i for i in moles_ref.columns if not i == oxygen_index]]
        .sum()
        .sum()
    )
    if not oxygen_constrained:  # oxygen unquantified, try to calculate using cations
        scale = ideal_cations / moles_cations
    else:  # oxygen quantified, try to calculate using oxygen
        scale = ideal_oxygens / moles_O

    moles_ref *= scale
    return moles_ref.sum(axis=0)
