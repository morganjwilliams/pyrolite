"""
CIPW Norm
===========

The CIPW (W. Cross, J. P. Iddings, L. V. Pirsson, and H. S. Washington) Norm was
introducted as a standard procedure for the estimation of rock-forming mineral
assemblages of igneous rocks from their geochemical compositions [Cross1902]_ .
This estimation process enables the approximate classificaiton of
microcrystalline and partially crystalline rocks using a range of
mineralogically-based classificaiton systems (e.g. most IUGS classifications),
and the generation of normative-mineral modifiers for geochemical classificaiton
systems.

A range of updated, modified and adjusted Norms were published in the century
following the original publication of the CIPW Norm, largely culminating in
Surendra Verma's 2003 paper "A revised CIPW norm" which enumerates an
algorithm for the estimation of an anhydrous Standard Igenous Norm (SIN)
[Verma2003]_ .
This was subsequently updated with the publication of IgRoCS [Verma2013]_ .
A version of this algorithm has now been implemented in
:mod:`pyrolite` (:func:`~pyrolite.mineral.normative.CIPW_norm`), and an overview
of the implementation and the currently available options is given below.

.. warning:: The current implementation of CIPW in pyrolite produces results
    which are inconsistent with SINCLAS/IgRoCS; we're working on addressing this.
    There's a warning implemented in the function so that you should be notified
    of this.

For the purposes of testing, pyrolite includes a file containing the outputs from
Verma's SINCLAS/IgRoCS program. Here we can use this file to demonstrate the use
of the CIPW Norm and verify that the results should generally be comparable
between Verma's original implementation and the :mod:`pyrolite` implementation.
Here we import this file and do a little cleaning and registration of
geochemical components so we can work with it in the sections to follow:
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrolite.geochem
from pyrolite.util.meta import pyrolite_datafolder

# sphinx_gallery_thumbnail_number = 2

df = (
    pd.read_csv(pyrolite_datafolder() / "testing" / "CIPW_Verma_Test.csv")
    .dropna(how="all", axis=1)
    .pyrochem.parse_chem()
)
df.pyrochem.compositional = df.pyrochem.compositional.apply(
    pd.to_numeric, errors="coerce"
).fillna(0)
df.loc[:, [c for c in df.columns if "NORM" in c]] = df.loc[
    :, [c for c in df.columns if "NORM" in c]
].apply(pd.to_numeric, errors="coerce")
########################################################################################
# The CIPW Norm can be accessed via :func:`pyrolite.mineral.normative.CIPW_norm`,
# and expects a dataframe as input containing major element oxides (in wt%) and
# can also use a select set of trace elements (in ppm).
#
from pyrolite.mineral.normative import CIPW_norm

NORM = CIPW_norm(df.pyrochem.compositional)
########################################################################################
# We can quickly check that this includes mineralogical data:
#
NORM.columns
########################################################################################
# The function accepts a few keyword arguments, all to do with the iron compositions
# and related adjustment/corrections:
#
# :code:`Fe_correction = <callable>`
#   For specifying the Fe-correction method/function. Currently only Le LeMaitre's
#   correction method is implemented [LeMaitre1976]_ .
#
# :code:`Fe_correction_mode = 'volcanic'`
#   For specificying the Fe-correction mode where relevant
#
# :code:`adjust_all_Fe = False`
#   Specifying whether you want to adjust all iron compositions, or only those
#   which are partially specified (i.e. only have a singular value for one of
#   FeO, Fe2O3, FeOT, Fe2O3T).
#
# For the purpose of establishing the congruency of our algorithm with Verma's,
# we'll use :code:`adjust_all_Fe = True`. Notably, this won't make too much
# difference to the format of the output, but it will adjust the estimates of
# normative mineralogy depending on oxidation state.
#
NORM = CIPW_norm(df.pyrochem.compositional, adjust_all_Fe=True)
########################################################################################
# Now we have the normative mineralogical outputs, we can have a look to see how
# these compare to some relevant geochemical inputs:
#
ax = NORM[["ilmenite", "magnetite"]].pyroplot.scatter(clip_on=False, c=df["TiO2"])
plt.show()
########################################################################################
ax = NORM[["orthoclase", "albite", "anorthite"]].pyroplot.scatter(
    clip_on=False, c=df["K2O"]
)
plt.show()
########################################################################################
# Coherency with SINCLAS / IgRoCS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Given we're reproducting an existing algorithm, it's prudent to check how closely
# the results match for a specific dataset to check whether there might be any numerical
# or computational errors. Below we go through this exercise for the test dataset we
# loaded above (which already includes the output of SINCLAS), comparing the original
# software to the pyrolite implementation.
#
# .. note:: Currently there are inconsistent results for a small number or samples
#   (deviations colormapped, and inconsistent results shown in red below), likely related
#   to the handling of iron components and their conversion.
#
# The output of SINCLAS has slightly different column naming that that of
# :mod:`pyrolite`, which provides full mineral names in the output dataframe
# columns. For this reason, we'll need to translate our NORM output columns
# to the SINCLAS column names. For this we can use the dictionary of minerals
# used in the CIPW Norm (:const:`~pyrolite.mineral.normative.NORM_MINERALS`)
from pyrolite.mineral.normative import NORM_MINERALS

translation = {
    d["name"]: (d.get("SINCLAS_abbrv", None) or k.upper()) + "_NORM"
    for k, d in NORM_MINERALS.items()
    if (d.get("SINCLAS_abbrv", None) or k.upper()) + "_NORM" in df.columns
}
translation
########################################################################################
# First we'll collect the minerals which appear in both dataframes, and then iterate
# through these to check how close the implementations are.
#
minerals = {
    k: v for (k, v) in translation.items() if (df[v] > 0).sum() and (NORM[k] > 0).sum()
}
########################################################################################
# To compare SINCLAS and the :mod:`pyrolite` NORM outputs, we'll construct a grid
# of plots which compare the respective mineralogical norms relative to a 1:1 line,
# and highlight discrepancies. As we'll do it twice below (once for samples labelled as
# volanic, and once for everything else), we may as well make a function of it.
#
# After that, let's take a look at the volcanic samples in isolation, which are the
# the key samples for which the NORM should be applied:
#
from pyrolite.plot.color import process_color


def compare_NORMs(SINCLAS_outputs, NORM_outputs, name=""):
    """
    Create a grid of axes comparing the outputs of SINCLAS and `pyrolite`'s NORM,
    after translating the column names to the appropriate form.
    """
    ncols = 4
    nrows = len(minerals.keys()) // ncols + 1 if len(minerals.keys()) % ncols else 0

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2))
    fig.suptitle(
        " - ".join(
            ["Comparing pyrolite's CIPW Norm to SINCLAS/IgRoCS"] + [name]
            if name
            else []
        ),
        fontsize=16,
        y=1.01,
    )
    ax = ax.flat
    for ix, (b, a) in enumerate(minerals.items()):
        ax[ix].set_title("\n".join(b.split()), y=0.9, va="top")
        if a in SINCLAS_outputs.columns and b in NORM_outputs.columns:
            # colour by deviation from unity
            c = process_color(
                np.abs((SINCLAS_outputs[a] / NORM_outputs[b]) - 1),
                cmap="RdBu_r",
                norm=plt.Normalize(vmin=0, vmax=0.1),
            )["c"]
            ax[ix].scatter(SINCLAS_outputs[a], NORM_outputs[b], c=c)
        # add a 1:1 line
        ax[ix].plot(
            [0, SINCLAS_outputs[a].max()],
            [0, SINCLAS_outputs[a].max()],
            color="k",
            ls="--",
        )

    for a in ax:
        a.set(xticks=[], yticks=[])  # turn off the ticks
        if not a.collections:  # turn off the axis for empty axes
            a.axis("off")
    return fig, ax


volcanic_filter = df.loc[:, "ROCK_TYPE"].str.lower().str.startswith("volc")
fig, ax = compare_NORMs(df.loc[volcanic_filter, :], NORM.loc[volcanic_filter])


########################################################################################
# And everything else:
#
fig, ax = compare_NORMs(df.loc[~volcanic_filter, :], NORM.loc[~volcanic_filter])
plt.show()
########################################################################################
# These normative mineralogical components could be input into mineralogical
# classifiers, as mentioned above. For example, the IUGS QAP classifier:
#
from pyrolite.util.classification import QAP

clf = QAP()  # build a QAP classifier

qap_data = NORM.loc[:, ["quartz", "orthoclase"]]  #
qap_data["plagioclase"] = NORM.loc[:, ["albite", "anorthite"]].sum(axis=1)
# predict which lithological class each mineralogical composiiton belongs in
# we add a small value to zeros here to ensure points fit in polygons
predicted_classes = clf.predict(qap_data.replace(0, 10e-6).values)
predicted_classes.head()
########################################################################################
# We can use these predicted classes as a color index also, within the QAP diagram
# or elsewhere:
#
ax = clf.add_to_axes()
qap_data.pyroplot.scatter(ax=ax, c=predicted_classes, axlabels=False, cmap="tab20c")
plt.show()
########################################################################################
# We could also compare how these mineralogical distinctions map into chemical ones
# like the TAS diagram:
#
from pyrolite.plot.templates import TAS

ax = TAS()
components = df.loc[:, ["SiO2"]]
components["alkali"] = df.loc[:, ["Na2O", "K2O"]].sum(axis=1)
# add the predictions from normative mineralogy to the TAS diagram
components.pyroplot.scatter(ax=ax, c=predicted_classes, cmap="tab20c", axlabels=False)
plt.show()
########################################################################################
# References
# ~~~~~~~~~~
#
# .. [Cross1902] Cross, W., Iddings, J. P., Pirsson, L. V., &
#     Washington, H. S. (1902).
#     A Quantitative Chemico-Mineralogical Classification and Nomenclature of
#     Igneous Rocks. The Journal of Geology, 10(6), 555–690.
#     `doi: 10.1086/621030 <https://doi.org/10.1086/621030>`__
#
# .. [Verma2003] Verma, S. P., Torres-Alvarado, I. S., & Velasco-Tapia, F. (2003).
#     A revised CIPW norm.
#     Swiss Bulletin of Mineralogy and Petrology, 83(2), 197–216.
#
# .. [Verma2013] Verma, S. P., & Rivera-Gomez, M. A. (2013). Computer Programs
#     for the Classification and Nomenclature of Igneous Rocks.
#     Episodes, 36(2), 115–124.
#
# .. [LeMaitre1976] Le Maitre, R. W (1976).
#     Some Problems of the Projection of Chemical Data into Mineralogical
#     Classifications.
#     Contributions to Mineralogy and Petrology 56, no. 2 (1 January 1976): 181–89.
#     `doi: doi.org/10.1007/BF00399603 <https://doi.org/10.1007/BF00399603>`__
#
