import pandas as pd
import pandas_flavor as pf
import numpy as np
import periodictable as pt
from pyrolite.geochem import to_molecular
from pyrolite.comp.codata import renormalise
from pyrolite.util.pd import to_frame
from collections import Counter, OrderedDict
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class Site(object):
    """
    Class for specifying mineral sites, including coordination information.

    Will be used for partitioning and affinity calculations for estimating mineral
    site chemistry.
    """

    def __init__(self, name, coordination=0, affinities={}):
        self.name = name
        self.coordination = coordination
        self.affinities = affinities

    def __str__(self):

        if self.coordination:
            return """{} Site `{}` with {}-fold coordination""".format(
                self.__class__.__name__, self.name, self.coordination
            )
        else:
            return """{} Site `{}`""".format(self.__class__.__name__, self.name)

    def __repr__(self):
        if self.coordination:
            return """{}("{}", {})""".format(
                self.__class__.__name__, self.name, self.coordination
            )
        else:
            return """{}("{}")""".format(self.__class__.__name__, self.name)

    def __eq__(self, other):
        pretest = (
            hasattr(other, "name")
            & hasattr(other, "coordination")
            & hasattr(other, "affinities")
        )
        if pretest:
            conds = (
                (self.__class__.__name__ == other.__class__.__name__)
                & (self.name == other.name)
                & (self.coordination == other.coordination)
                & (self.affinities == other.affinities)
            )

            return conds
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__().encode("UTF-8"))


class MX(Site):
    """
    Octahedrally coordinated M site.
    """

    def __init__(self, name="M", coordination=8, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)


class TX(Site):
    """
    Tetrahedrally coordinated T site.
    """

    def __init__(self, name="T", coordination=4, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)


class IX(Site):
    """
    Dodecahedrally coordinated I site.
    """

    def __init__(self, name="I", coordination=12, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)


class VX(Site):
    """
    Vacancy site.
    """

    def __init__(self, name="V", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)


class OX(Site):
    """
    Oxygen site.
    """

    def __init__(self, name="O", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)


class AX(Site):
    """
    Anion site.
    """

    def __init__(self, name="A", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)


class MineralStructure(object):
    def __init__(self, name, *components):
        self.name = name
        self.structure = None
        self.set_structure(*components)

    def set_structure(self, *components):
        if len(components):
            self.components = list(components)
            _bag = []
            for c in self.components:
                if c not in _bag:
                    _bag.append(c)
            self.structure = OrderedDict()
            for item in _bag:
                self.structure[item] = self.components.count(item)

    def __repr__(self):
        if self.structure is not None:
            component_string = ", ".join(
                ["{}".format(c.__repr__()) for c in self.structure.keys()]
            )
            return """{}("{}", {})""".format(
                self.__class__.__name__, self.name, component_string
            )
        else:
            return """{}("{}")""".format(self.__class__.__name__, self.name)

    def __str__(self):
        if self.structure is not None:
            c_list = []
            names = [c.name for c in self.structure.keys()]
            counts = [self.structure[c] for c in self.structure.keys()]
            for site in self.structure.keys():
                n, c = site.name, self.structure[site]
                if c > 1:
                    c_str = "[{}]$_{}$".format(n, c)
                else:
                    c_str = "[{}]".format(n)
                c_list.append(c_str)
            component_string = "".join(c_list)
            return """{} {}""".format(self.name, component_string)
        else:
            return """{}""".format(self.name)

    def __hash__(self):
        return hash(self.__repr__().encode("UTF-8"))


class TSite(TX):
    def __init__(self):
        super().__init__(self, affinities={"Si{4+}": 0, "Al{3+}": 1, "Fe{3+}": 2})


@pf.register_series_method
@pf.register_dataframe_method
def recalculate_cations(
    df, ideal_cations=4, ideal_oxygens=6, Fe_species=["FeO", "Fe", "Fe2O3"]
):
    """
    Recalculate a composition to a.p.f.u.
    """

    # if Fe2O3 and FeO are specified, calculate based on oxygen

    moles = to_frame(df).to_molecular(renorm=False)
    moles = moles.where(~np.isclose(moles, 0.0), np.nan)

    count_iron_species = np.array([i in moles.columns for i in Fe_species]).sum()
    if count_iron_species > 1:  # check that only one is defined
        use_oxygen = (
            count_iron_species - pd.isnull(moles.loc[:, Fe_species]).all(axis=1).sum()
        ) > 1

        if use_oxygen:
            print("using oxygen")
            logger.info("Multiple iron species defined. Calculating using oxygen.")
        else:
            logger.info("Single iron species defined. Calculating using cations.")

    components = moles.columns
    if use_oxygen:  # need to specifically separate Fe2 and Fe3
        parts = [
            {str(k): v for (k, v) in pt.formula(c).atoms.items()} for c in components
        ]

        schema = []
        for p in parts:
            oxygens = p["O"]
            other_components = [i for i in p.keys() if not i == "O"]
            assert len(other_components) == 1  # need to be simple oxides
            other = other_components[0]
            charge = oxygens * 2 / p[other]
            ion = pt.formula("%s{%i+}" % (other, int(charge)))
            schema.append({str(ion): p[other], "O": oxygens})

    else:
        schema = [pt.formula(c).atoms for c in components]
    ref = pd.DataFrame(data=schema)
    ref.index = components
    cation_masses = {c: pt.formula(c).mass for c in ref.columns}
    ref.columns = ref.columns.map(str)
    ref = ref.loc[:, [i for i in ref.columns if not i == "O"] + ["O"]]
    moles_ref = ref.copy()
    moles_ref.loc[:, :] = ref.values * moles.T.values
    moles_O = moles_ref["O"].sum()
    moles_cations = (
        moles_ref.loc[:, [i for i in moles_ref.columns if not i == "O"]].sum().sum()
    )
    if not use_oxygen:  # oxygen unquantified, try to calculate using cations
        scale = ideal_cations / moles_cations
    else:  # oxygen quantified, try to calculate using oxygen
        scale = ideal_oxygens / moles_O

    moles_ref *= scale
    return moles_ref


def recalculate_pyroxene(df):
    """
    Mineral Recalculation for Pyroxene

    Todo
    ------
    * Endmember separation
    * Site assignment
    * Checks and balances within margins
    """
    moles_ref = recalculate_cations(df, ideal_cations=4, ideal_oxygens=6)
    # Assignment
    # Si, Al assigned to tetrahedral site
    # Ca, Na, Mn, Fe2, Mg assigned to M2
    # Fe3, Ti, Al, Fe2, Mg assigned to M1
    # Fe2/Mg assumed the same in M1-M2
    return moles_ref


def recalculate_olivine(df):
    """
    Mineral Recalculation for Olivines

    Todo
    ------
    * Endmember separation
    * Site assignment
    * Checks and balances within margins
    """
    pass


# %% ---
ox, vals = (
    "SiO2, TiO2, Al2O3, Cr2O3, Fe2O3, FeO, MnO, MgO, CaO, Na2O".split(", "),
    [
        float(i)
        for i in "63.338	0.000	3.231	0.001	0.001	12.339	0.000	19.823	1.269	0.000".split(
            "\t"
        )
    ],
)
pyxdf = pd.Series(data=np.array(vals), index=ox)

recalculate_pyroxene(pyxdf)

olivine = MineralStructure("olivine", MX("M1"), MX("M2"), TX(), OX())
pyroxene = MineralStructure(
    "pyroxene",
    MX("M1", affinities={}),
    MX("M2"),
    TX(affinities={"Si{4+}": 0, "Al{3+}": 1, "Fe{3+}": 2}),
    TX(affinities={"Si{4+}": 0, "Al{3+}": 1, "Fe{3+}": 2}),
    OX(),
)
pyroxene
str(pyroxene)


if 0:
    print("a")

M1 = OX()
print(M1)
O = pt.O
