import pandas as pd
import pandas_flavor as pf
import numpy as np
import periodictable as pt
from pyrolite.comp.codata import renormalise
from pyrolite.util.pd import to_frame
from collections import OrderedDict
from scipy.optimize import least_squares, Bounds
from .sites import *
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class MineralTemplate(object):
    """
    Generic mineral stucture template. Formatted collection of crystallographic sites.
    """

    def __init__(self, name, *components):
        self.name = name
        self.structure = None
        self.site_occupancy = None
        self.set_structure(*components)

    def set_structure(self, *components):
        self.components = list(components)
        self.structure = OrderedDict()
        if len(components):
            _bag = []
            for c in self.components:
                if not isinstance(c, Site):
                    c = Site(c)
                if c not in _bag:
                    _bag.append(c)
            for item in _bag:
                self.structure[item] = self.components.count(item)

        self.affinities = {c: c.affinities for c in self.components}
        self.ideal_cations = sum([c.cationic for c in self.components])
        self.ideal_oxygens = sum([c.oxygen for c in self.components])

    def copy(self):
        return MineralTemplate(self.name, *self.components)

    def __repr__(self):
        if self.structure is not None:
            component_string = ", ".join(
                ["{}".format(c.__repr__()) for c in list(self.structure)]
            )
            return """{}("{}", {})""".format(
                self.__class__.__name__, self.name, component_string
            )
        else:
            return """{}("{}")""".format(self.__class__.__name__, self.name)

    def __str__(self):
        if self.structure is not None:
            structure = self.structure
            c_list = []
            names = [c.name for c in list(structure)]
            counts = [structure[c] for c in list(structure)]
            for site in list(structure):
                n, c = site.name, structure[site]
                if c > 1:
                    c_str = "[{}]$_{}$".format(n, c)
                else:
                    c_str = "[{}]".format(n)
                c_list.append(c_str)
            component_string = "".join(c_list)
            return """`{}` {}""".format(self.name, component_string)
        else:
            return """`{}`""".format(self.name)

    def __hash__(self):
        return hash(self.__repr__().encode("UTF-8"))


class Mineral(object):
    """Mineral, with structure and composition."""

    def __init__(self, name=None, template=None, composition=None, endmembers=None):
        self.name = name
        self.endmembers = endmembers
        self.template = None
        self.composition = None
        self.set_template(template)
        self.set_composition(composition)
        self.endmember_decomposition = None

        self.init = True

    def set_template(self, template, name=None):
        """Assign a mineral template to the mineral."""
        if template is not None:
            if name is None:
                name = self.name
            if not isinstance(template, MineralTemplate):
                template = MineralTemplate(name, *template)
            else:
                template = template.copy()
        else:
            template = MineralTemplate("")
        if template is not None:
            logger.debug("Setting Template: {}".format(template))
        else:
            logger.debug("Clearing Template")
        self.template = template
        self.sites = [i for i in list(self.template.structure)]
        self.recalculate_cations()

    def set_composition(self, composition):
        """Assign a composition to the mineral."""
        if composition is not None:
            if isinstance(composition, pd.Series):
                # convert to molecular oxides, then to formula, then to wt% elemental
                components = [pt.formula(c) for c in composition.index]
                values = composition.values

                formula = heteromolecule(
                    [v / c.mass * c for v, c in zip(values, components)]
                )
                composition = pd.Series(formula_to_elemental(formula))
            elif isinstance(composition, pt.formulas.Formula):
                composition = pd.Series(formula_to_elemental(composition))
            else:
                composition = pd.Series(composition)
                # convert composition to elemental

            logger.debug(
                "Setting Composition: {}".format(
                    {k: np.round(v, 4) for k, v in composition.to_dict().items()}
                )
            )
        else:
            logger.debug("Clearing Composition")

        self.composition = composition
        self.recalculate_cations()

    def recalculate_cations(
        self,
        composition=None,
        ideal_cations=None,
        ideal_oxygens=None,
        Fe_species=["FeO", "Fe", "Fe2O3"],
    ):
        composition = composition or self.composition

        if composition is not None:
            ideal_cations = ideal_cations or self.template.ideal_cations
            ideal_oxygens = ideal_oxygens or self.template.ideal_cations

            self.cationic_composition = recalc_cations(
                self.composition,
                ideal_cations=ideal_cations,
                ideal_oxygens=ideal_oxygens,
            )
            return self.cationic_composition

    def apfu(self):
        """
        Get the atoms per formula unit.
        """
        return self.recalculate_cations()

    def endmember_decompose(self, det_lim=0.01):
        """
        Decompose a mineral composition into endmember components.
        """
        assert self.endmembers is not None

        # take endmembers with components which may be present in composition
        _target_components = set(self.composition.index.values)
        potential_components = []
        for em, tem in self.endmembers.items():
            _components = set(tem.composition.index.values)
            if _components.issubset(_target_components):
                potential_components.append((em, tem.composition))

        compositions = pd.concat(
            [c for em, c in potential_components], axis=1, sort=False
        ).fillna(0)
        compositions.columns = [em for em, c in potential_components]
        weights = np.ones((compositions.columns.size))
        weights /= weights.sum()

        x = compositions.values.T
        y = self.composition.reindex(compositions.index).fillna(0).values

        def mixture(weights, x, y):
            return weights @ x - y

        res = least_squares(
            mixture,
            weights,
            bounds=([0.0] * weights.shape[0], [1.0] * weights.shape[0]),
            args=(x, y),
        )
        abundances, cost = res.x, res.cost
        if cost > det_lim:
            logger.warn("Residuals are higher than detection limits.")

        abundances[
            (np.isclose(abundances, 0.0, atol=1e-06) | (abundances <= det_lim))
        ] = np.nan
        abundances /= np.nansum(abundances)
        self.endmember_decomposition = {
            c: v for (c, v) in zip(compositions.columns, abundances) if not np.isnan(v)
        }
        # optimise decomposition into endmember components
        return self.endmember_decomposition

    def calculate_occupancy(
        self, composition=None, error=10e-6, balances=[["Fe{2+}", "Mg{2+}"]]
    ):
        """
        Calculate the estimated site occupancy for a given composition.
        Ions will be assigned to sites according to affinities. Sites with equal
        affinities should recieve equal assignment.
        """
        if self.template is not None:
            self.recalculate_cations()

            composition = composition or self.cationic_composition

            affinities = pd.DataFrame(
                [site.affinities for site in self.template.structure]
            ).T
            affinities.columns = self.sites

            occupancy = affinities.copy().reindex(composition.index)

            unknown_site_ions = occupancy.loc[
                occupancy.count(axis=1) == 0, :
            ].index.values

            if len(unknown_site_ions):
                logging.warn("Unknown site for: {}".format(unknown_site_ions))

            occupancy.loc[:, :] = 0.0

            for site in self.sites:
                site.occupancy = pd.Series(index=occupancy.index).fillna(0)

            inventory = composition.copy()
            for site in self.sites[::-1]:
                accepts = [
                    i
                    for i in sorted(site.affinities, key=site.affinities.__getitem__)
                    if i in inventory.index
                ]
                capacity = np.float(self.template.structure[site])
                site_balances = [b for b in balances if all([i in accepts for i in b])]
                direct_assign = [
                    i for i in accepts if not any([i in b for b in site_balances])
                ]

                for ion in direct_assign:
                    current = site.occupancy.sum()
                    if not np.isclose(current, capacity + error):
                        assigning = np.nanmin([capacity - current, inventory[ion]])
                        if not assigning + current - (capacity + error) > 0.0:
                            logger.debug(
                                "Assigning {:.3f} {} to Site {}".format(
                                    assigning, ion, site
                                )
                            )
                            occupancy.loc[ion, site] += assigning
                            site.occupancy[ion] += occupancy.loc[ion, site]
                            inventory[ion] -= assigning
                        else:
                            logger.warn(
                                "{} capacity encountered: {} / {}".format(
                                    site, assigning + current, capacity
                                )
                            )

                for group in site_balances:
                    current = site.occupancy.sum()
                    invent = inventory.loc[group].sum()
                    fractions = inventory.loc[group] / inventory.loc[group].sum()
                    if not np.isclose(current, capacity + error):
                        assigning = np.nanmin([capacity - current, invent])
                        if not assigning + current - (capacity + error) > 0.0:
                            logger.debug(
                                "Assigning {:.3f} {} to Site {}".format(
                                    assigning, ion, site
                                )
                            )
                            assigning *= fractions
                            occupancy.loc[group, site] += assigning
                            site.occupancy[group] += occupancy.loc[group, site]
                            inventory.loc[group] -= assigning
                        else:
                            logger.warn(
                                "{} capacity encountered: {} / {}".format(
                                    site, assigning + current, capacity
                                )
                            )

            # check sums across all sites equal the full composition

            self.template.site_occupancy = occupancy
            return occupancy

    def get_site_occupancy(self):
        self.calculate_occupancy()
        return self.template.site_occupancy

    def __format__(self, format_spec=":s"):
        return str(self)

    def __str__(self):
        D = {}
        for kwarg in ["name", "template"]:
            val = getattr(self, kwarg, None)
            if val is not None:
                D[kwarg] = val
        callstrings = []
        for k, v in D.items():
            callstrings.append("""{}""".format(v.__str__()))

        strstring = r"""{}: """.format(self.__class__.__name__) + ", ".join(callstrings)
        return strstring

    def __repr__(self):
        D = {}
        for kwarg in ["name", "template", "endmembers"]:
            val = getattr(self, kwarg, None)
            if val is not None:
                D[kwarg] = val

        callstrings = []
        for k, v in D.items():
            callstrings.append("""{}={},""".format(k, v.__repr__()))

        reprstring = (
            r"""{}(""".format(self.__class__.__name__) + "".join(callstrings) + r""")"""
        )
        return reprstring


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


def heteromolecule(formulas):
    """Combine multiple formulas into one."""

    molecule = pt.formula("")
    for f in formulas:
        molecule += pt.formula(f)
    return molecule


@pf.register_series_method
@pf.register_dataframe_method
def recalc_cations(
    df, ideal_cations=4, ideal_oxygens=6, Fe_species=["FeO", "Fe", "Fe2O3"]
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
    count_iron_species = np.array([i in moles.columns for i in Fe_species]).sum()
    oxygen_constrained = False
    if count_iron_species > 1:  # check that only one is defined
        oxygen_constrained = (
            count_iron_species - pd.isnull(moles.loc[:, Fe_species]).all(axis=1).sum()
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
        parts = [
            {str(k): v for (k, v) in pt.formula(c).atoms.items()} for c in components
        ]
        for p in parts:
            oxygens = p["O"]
            other_components = [i for i in list(p) if not i == "O"]
            assert len(other_components) == 1  # need to be simple oxides
            other = other_components[0]
            charge = oxygens * 2 / p[other]
            ion = pt.formula("%s{%i+}" % (other, int(charge)))
            schema.append({str(ion): p[other], "O": oxygens})
    else:
        # elemental composition
        parts = components
        for part in parts:
            p = list(pt.formula(part).atoms)[0]
            if p.charge != 0:
                charge = p.charge
            else:
                p.charge = p.default_charge
            schema.append({str(p): 1})

    ref = pd.DataFrame(data=schema)
    ref.index = components
    cation_masses = {c: pt.formula(c).mass for c in ref.columns}
    oxygen_index = [i for i in ref.columns if "O" in i][0]
    ref = ref.loc[:, [i for i in ref.columns if not i == oxygen_index] + [oxygen_index]]
    moles_ref = ref.copy()

    moles_ref.loc[:, :] = ref.values * moles.T.values

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
