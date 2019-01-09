from .mineral import *
from .sites import *
# %% Generic Mineral Group Templates ---------------------------------------------------
olivine = MineralTemplate(
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

pyroxene = MineralTemplate(
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

# %% Specific Minerals -----------------------------------------------------------------
forsterite = Mineral("forsterite", olivine, pt.formula("Mg2SiO4"))
fayalite = Mineral("fayalite", olivine, pt.formula("Fe2SiO4"))
tephroite = Mineral("tephroite", olivine, pt.formula("Mn2SiO4"))
liebenbergite = Mineral("liebenbergite", olivine, pt.formula("Ni1.5Mg0.5SiO4"))

enstatite = Mineral("enstatite", pyroxene, pt.formula("Mg2Si2O6"))
ferrosilite = Mineral("ferrosilite", pyroxene, pt.formula("Fe2Si2O6"))
diopside = Mineral("diopside", pyroxene, pt.formula("CaMgSi2O6"))
hedenbergite = Mineral("hedenbergite", pyroxene, pt.formula("CaFeSi2O6"))
johannsenite = Mineral("johannsenite", pyroxene, pt.formula("CaMnSi2O6"))
esseneite = Mineral("esseneite", pyroxene, pt.formula("Ca Al Fe{3+} SiO6"))
jadeite = Mineral("jadeite", pyroxene, pt.formula("NaAlSi2O6"))
aegirine = Mineral("aegirine", pyroxene, pt.formula("NaFe{3+}Si2O6"))
namansilite = Mineral("namansilite", pyroxene, pt.formula("NaMn{3+}Si2O6"))
kosmochlor = Mineral("kosmochlor", pyroxene, pt.formula("NaCrSi2O6"))
spodumene = Mineral("spodumene", pyroxene, pt.formula("LiAlSi2O6"))
