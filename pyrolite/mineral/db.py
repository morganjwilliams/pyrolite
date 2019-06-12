"""
Rudimentary draft mineral database.

Todo
----

    * This is one of the major bottlenecks for package import, due to the execution.

        Will be able to clean this up with either delayed execution or using a database.
"""
import periodictable as pt
from .mineral import *
from .sites import *

# %% Generic Mineral Group Templates ---------------------------------------------------
OLIVINE = MineralTemplate(
    "olivine",
    MX(
        "M1",
        affinities={
            "Mg{2+}": 0,
            "Fe{2+}": 1,
            "Mn{2+}": 2,
            "Li{+}": 3,
            "Ca{2+}": 4,
            "Na{+}": 5,
        },
    ),
    MX(
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
    ),
    TX(),
    *[OX()] * 2,
)


Mineral("forsterite", OLIVINE, pt.formula("Mg2SiO4"))
Mineral("fayalite", OLIVINE, pt.formula("Fe2SiO4"))
Mineral("tephroite", OLIVINE, pt.formula("Mn2SiO4"))
Mineral("liebenbergite", OLIVINE, pt.formula("Ni1.5Mg0.5SiO4"))

PYROXENE = MineralTemplate(
    "pyroxene",
    MX(
        "M1",
        affinities={
            "Mg{2+}": 0,
            "Fe{2+}": 1,
            "Mn{2+}": 2,
            "Li{+}": 3,
            "Ca{2+}": 4,
            "Na{+}": 5,
        },
    ),
    MX(
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
    ),
    *[TX()] * 2,
    *[OX()] * 6,
)


Mineral("enstatite", PYROXENE, pt.formula("Mg2Si2O6"))
Mineral("ferrosilite", PYROXENE, pt.formula("Fe2Si2O6"))
Mineral("diopside", PYROXENE, pt.formula("CaMgSi2O6"))
Mineral("hedenbergite", PYROXENE, pt.formula("CaFeSi2O6"))
Mineral("johannsenite", PYROXENE, pt.formula("CaMnSi2O6"))
Mineral("esseneite", PYROXENE, pt.formula("Ca Al Fe{3+} SiO6"))
Mineral("jadeite", PYROXENE, pt.formula("NaAlSi2O6"))
Mineral("aegirine", PYROXENE, pt.formula("NaFe{3+}Si2O6"))
Mineral("namansilite", PYROXENE, pt.formula("NaMn{3+}Si2O6"))
Mineral("kosmochlor", PYROXENE, pt.formula("NaCrSi2O6"))
Mineral("spodumene", PYROXENE, pt.formula("LiAlSi2O6"))


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

Mineral("spinel", SPINEL, pt.formula("MgAl2O4"))
Mineral("magnesioferrite", SPINEL, pt.formula("MgFe{3+}2O4"))
Mineral("magnesiochromite", SPINEL, pt.formula("MgCr{3+}2O4"))
Mineral("magnetite", SPINEL, pt.formula("Fe{2+}Fe{3+}2O4"))
Mineral("hercynite", SPINEL, pt.formula("Fe{2+}Al2O4"))
Mineral("chromite", SPINEL, pt.formula("Fe{2+}Cr{3+}2O4"))
