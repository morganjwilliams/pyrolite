import pandas as pd
import numpy as np
import periodictable as pt
from collections import OrderedDict
import scipy.optimize
from .sites import Site, MX, TX, OX
from .transform import recalc_cations
from .mindb import get_mineral, parse_composition
from ..util.log import Handle

logger = Handle(__name__)


class MineralTemplate(object):
    def __init__(self, name, *components):
        """
        Generic mineral stucture template. Formatted collection of crystallographic sites.
        """
        self.name = name
        self.structure = {}
        self.site_occupancy = None
        self.set_structure(*components)

    def set_structure(self, *components):
        """
        Set the structure of the mineral template.

        Parameters
        ----------
        components
            Argument list consisting of each of the structural components. Can consist
            of any mixture of Sites or argument tuples which can be passed to
            Site __init__.
        """
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

        self.affinities = {c: c.affinities for c in self.structure}
        self.ideal_cations = sum(
            [c.cationic * self.structure[c] for c in self.structure]
        )
        self.ideal_oxygens = sum([c.oxygen * self.structure[c] for c in self.structure])

    def copy(self):
        return MineralTemplate(self.name, *self.components)

    def __repr__(self):
        if self.structure != {}:
            component_string = ", ".join(
                ["{}".format(c.__repr__()) for c in list(self.structure)]
            )
            return """{}("{}", {})""".format(
                self.__class__.__name__, self.name, component_string
            )
        else:
            return """{}("{}")""".format(self.__class__.__name__, self.name)

    def __str__(self):
        if self.structure != {}:
            structure = self.structure
            c_list = []
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
    def __init__(self, name=None, template=None, composition=None, endmembers=None):
        """Mineral, with structure and composition."""
        self.name = name
        self.template = None
        self.composition = None
        self.formula = None
        self.endmembers = {}
        self.set_template(template)
        self.set_composition(composition)
        self.set_endmembers(endmembers)
        self.endmember_decomposition = None
        self.init = True

    def set_endmembers(self, endmembers=None):
        """Set the endmbmer components for a mineral."""
        if endmembers is not None:
            if isinstance(endmembers, list):
                for em in endmembers:
                    self.add_endmember(em)
            elif isinstance(endmembers, dict):
                for name, em in endmembers.items():
                    self.add_endmember(em, name=name)

    def add_endmember(self, em, name=None):
        """Add a single endmember to the database."""
        mineral = em
        if isinstance(mineral, tuple):
            name, mineral = mineral
        if mineral is not None:
            # process different options for getting a mineral output
            if isinstance(mineral, str):
                name = name or str(mineral)
                mineral = Mineral(name, None, pt.formula(get_mineral(em)["formula"]))
            elif isinstance(mineral, pt.formulas.Formula):
                name = name or str(mineral)
                mineral = Mineral(name, None, mineral)
            else:
                pass

            name = name or mineral.name
            self.endmembers[name] = mineral

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

    def set_composition(self, composition=None):
        """
        Parse and assign a composition to the mineral.

        Parameters
        ---------
        composition
            Composition to assign to the mineral. Can be provided in any form which is
            digestable by parse_composition.
        """
        if isinstance(composition, pt.formulas.Formula):
            self.formula = composition
        composition = parse_composition(composition)
        if composition is not None:
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
        oxygen_constrained=False,
    ):
        """
        Recalculate a composition to give an elemental ionic breakdown.

        Parameters
        ----------
        composition
            Composition to recalculate. If not provided, will try to use the mineral
            composition as set.
        ideal_cations : int
            Ideal number of cations to use for formulae calcuations. Will only be used
            if oxygen is constrained (i.e. multiple Fe species present or
            oxygen_constrained=True).
        ideal_oxygens : int
            Ideal number of oxygens to use for formulae calcuations. Will only be used
            if oxygen is not constrained (i.e. single Fe species present and
            oxygen_constrained=False).
        Fe_species : list
            List of iron species for identifying redox-defined compositions.
        oxygen_constrained : bool, False
            Whether the oxygen is a closed or open system for the specific composition.
        """
        composition = composition or self.composition

        if composition is not None:
            ideal_cations = ideal_cations or self.template.ideal_cations
            ideal_oxygens = ideal_oxygens or self.template.ideal_cations

            self.cationic_composition = recalc_cations(
                self.composition,
                ideal_cations=ideal_cations,
                ideal_oxygens=ideal_oxygens,
                Fe_species=Fe_species,
                oxygen_constrained=oxygen_constrained,
            )
            return self.cationic_composition

    def apfu(self):
        """Get the atoms per formula unit."""
        # recalculate_cations return apfu by default
        return self.recalculate_cations()

    def endmember_decompose(self, det_lim=0.01):
        """
        Decompose a mineral composition into endmember components.

        Parameters
        ----------
        det_lim : float
            Detection limit for individual

        Notes
        -----
        Currently implmented using optimization based on mass fractions.

        Todo
        -----
        Implement site-based endmember decomposition, which will enable more checks and
        balances.
        """
        assert self.endmembers is not None

        # take endmembers with components which may be present in composition
        _target_components = set(self.composition.index.values)
        potential_components = []
        for em, tem in self.endmembers.items():
            _components = set(tem.composition.index.values)
            if _components.issubset(_target_components):
                potential_components.append((em, tem))

        compositions = pd.concat(
            [c.composition for em, c in potential_components], axis=1, sort=False
        ).fillna(0)
        compositions.columns = [em for em, c in potential_components]
        weights = np.ones((compositions.columns.size))
        weights /= weights.sum()

        x = compositions.values.T
        y = self.composition.reindex(compositions.index).fillna(0).values

        def mixture(weights, x, y):
            return weights @ (x - y)

        res = scipy.optimize.least_squares(
            mixture,
            weights,
            bounds=([0.0] * weights.shape[0], [1.0] * weights.shape[0]),
            args=(x, y),
        )
        abundances, cost = res.x, res.cost
        if cost > det_lim:
            logger.warn("Residuals are higher than detection limits.")

        # convert abundances to molecular
        abundances = pd.Series(
            {c: v for (c, v) in zip(compositions.columns, abundances)}
        )
        abundances = abundances.div([c.formula.mass for em, c in potential_components])
        abundances = abundances.div(abundances.sum())
        abundances.loc[
            (np.isclose(abundances, 0.0, atol=1e-06) | (abundances <= det_lim))
        ] = np.nan
        abundances = abundances.loc[~pd.isnull(abundances)]
        abundances /= abundances.sum()
        # optimise decomposition into endmember components
        self.endmember_decomposition = abundances.to_dict()
        return self.endmember_decomposition

    def calculate_occupancy(
        self, composition=None, error=10e-6, balances=[["Fe{2+}", "Mg{2+}"]]
    ):
        """
        Calculate the estimated site occupancy for a given composition.
        Ions will be assigned to sites according to affinities. Sites with equal
        affinities should recieve equal assignment.

        Parameters
        -----------
        composition
            Composition to calculate site occupancy for.
        error : float
            Absolute error for floating point occupancy calculations.
        balances : list
            List of iterables containing ions to balance across multiple sites. Note
            that the partitioning will occur after non-balanced cations are assigned,
            and that ions are only balanced between sites which have defined affinities
            for all of the particular ions defined in the 'balance'.
        """
        if self.template is not None:
            if composition is None:
                self.recalculate_cations()
                composition = self.cationic_composition
            else:
                composition = parse_composition(composition)

            if composition is None:
                logger.warn("Composition not set. Cannot calculate occupancy.")

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
                site.occupancy = pd.Series(index=occupancy.index, dtype="float").fillna(
                    0
                )

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
        else:
            logger.warn("Template not yet set. Cannot calculate occupancy.")

    def get_site_occupancy(self):
        """Get the site occupancy for the mineral."""
        self.calculate_occupancy()
        return self.template.site_occupancy

    def __str__(self):
        """Get a string representation of the Mineral."""
        D = {}
        for kwarg in ["name", "template"]:
            val = getattr(self, kwarg, None)
            if val is not None:
                D[kwarg] = val
        callstrings = []
        for v in D.values():
            callstrings.append("""{}""".format(v.__str__()))

        strstring = r"""{}: """.format(self.__class__.__name__) + ", ".join(callstrings)
        return strstring

    def __repr__(self):
        """Get the call signature of the Mineral."""
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

    def __hash__(self):
        """Get the hash for the call signature of the Mineral."""
        return hash(self.__repr__().encode("UTF-8"))


M1 = MX(
    "M1",
    affinities={
        "Mg{2+}": 0,
        "Fe{2+}": 1,
        "Mn{2+}": 2,
        "Li{+}": 3,
        "Ca{2+}": 4,
        "Na{+}": 5,
    },
)
M2 = MX(
    "M2",
    affinities={
        "Al{3+}": 0,
        "Fe{3+}": 1,
        "Ti{4+}": 2,
        "Cr{3+}": 3,
        "V{3+}": 4,
        "Ti{3+}": 5,
        "Zr{4+}": 6,
        "Sc{3+}": 7,
        "Zn{2+}": 8,
        "Mg{2+}": 9,
        "Fe{2+}": 10,
        "Mn{2+}": 11,
    },
)
OLIVINE = MineralTemplate("olivine", M1, M2, TX(), *[OX()] * 2,)

PYROXENE = MineralTemplate("pyroxene", M1, M2, *[TX()] * 2, *[OX()] * 6,)

SPINEL = MineralTemplate(
    "spinel",
    Site(
        "A",
        affinities={"Mg{2+}": 0, "Fe{2+}": 1, "Mn{2+}": 2, "Zn{2+}": 3},
        coordination=4,
    ),
    *[
        Site(
            "B",
            affinities={"Al{3+}": 0, "Fe{3+}": 1, "Cr{3+}": 3, "V{3+}": 3},
            coordination=6,
        )
    ]
    * 2,
    *[OX()] * 4,
)
