import numpy as np
import pandas as pd
import ternary as pyternary
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
from pyrolite.comp.codata import *
from pyrolite.util.plot import (
    plot_pca_vectors,
    plot_stdev_ellipses,
    ternary_heatmap,
    plot_Z_percentiles,
    percentile_contour_values_from_meshz,
    bin_centres_to_edges,
    bin_edges_to_centres,
)
from pyrolite.util.skl import ILRTransform, ALRTransform
from pyrolite.util.synthetic import random_composition

np.random.seed(82)
# %% colors ----------------------------------------------------------------------------
t10b3 = [  # tableau 10 colorblind safe colors, a selection of 3
    (r / 255.0, g / 255.0, b / 255.0)
    for r, g, b in [(0, 107, 164), (171, 171, 171), (89, 89, 89), (95, 158, 209)]
]
# %% data and transforms ---------------------------------------------------------------
d = 1.0  # distance from centre
sig = 0.1  # scale for variance
# means for logspace (D=2)
means = np.array(np.meshgrid([-1, 1], [-1, 1])).T.reshape(-1, 2) * d
# means = np.array([(-d, -d), (d, -d), (-d, d), (d, d)])
covs = (  # covariance for logspace (D=2)
    np.array(
        [
            [[1, 0], [0, 1]],
            [[0.5, 0.15], [0.15, 0.5]],
            [[1.5, -1], [-1, 1.5]],
            [[1.2, -0.6], [-0.6, 1.2]],
        ]
    )
    * sig
)

means = ILRTransform().inverse_transform(means)  # compositional means (D=3)
size = 2000
pts = [random_composition(mean=M, cov=C, size=2000) for M, C in zip(means, covs)]

T = ILRTransform()
to_log = T.transform
from_log = T.inverse_transform

df = pd.DataFrame(np.vstack(pts))
df.columns = ["SiO2", "MgO", "FeO"]
df["Sample"] = np.repeat(np.arange(df.columns.size + 1), 2000).flatten()
# %% figure --
scale = 100
fig, ax = plt.subplots(2, 2, figsize=(10, 10 * np.sqrt(3) / 2))
ax = ax.flat
for a in ax:  # append ternary axes
    _, a.tax = pyternary.figure(ax=a, scale=scale)
    a.tax.boundary(linewidth=1.0)
# %% scatter ---------------------------------------------------------------------------
kwargs = dict(marker="D", alpha=0.1, s=3, no_ticks=True, axlabels=False)
for ix, sample in enumerate(df.Sample.unique()):
    comp = df.query("Sample == {}".format(sample)).loc[:, ["SiO2", "MgO", "FeO"]]
    comp.pyroplot.ternary(ax=ax[0], color=t10b3[ix], **kwargs)
# %% ellipses and vectors from PCA -----------------------------------------------------
kwargs = dict(ax=ax[1], transform=from_log, nstds=3)
for ix, sample in enumerate(df.Sample.unique()):
    comp = df.query("Sample == {}".format(sample)).loc[:, ["SiO2", "MgO", "FeO"]]
    tcomp = to_log(comp)
    plot_stdev_ellipses(tcomp.values, color=t10b3[ix], **kwargs)
    plot_pca_vectors(tcomp.values, ls="-", lw=0.5, color="k", **kwargs)
# %% individual density diagrams ------------------------------------------------------
kwargs = dict(ax=ax[-2], bins=100, no_ticks=True, axlabels=False)
for ix, sample in enumerate(df.Sample.unique()):
    comp = df.query("Sample == {}".format(sample)).loc[:, ["SiO2", "MgO", "FeO"]]
    comp.pyroplot.density(cmap="Blues", pcolor=True, **kwargs)
    comp.pyroplot.density(
        contours=[0.68, 0.95],
        cmap="Blues_r",
        contour_labels={0.68: "σ", 0.95: "2σ"},
        **kwargs,
    )
# %% overall density diagram ----------------------------------------------------------
kwargs = dict(ax=ax[-1], no_ticks=True, axlabels=False)
df.loc[:, ["SiO2", "MgO", "FeO"]].pyroplot.density(bins=100, cmap="Greys", **kwargs)
# %% axes cleanup
for a in ax:
    for side in ["top", "right", "bottom", "left"]:
        a.spines[side].set_visible(False)
    a.get_xaxis().set_ticks([])
    a.get_yaxis().set_ticks([])
    a.patch.set_facecolor(None)
    a.patch.set_visible(False)
    a.set_aspect("equal")

# %% Save Figure
from pyrolite.util.plot import save_axes, save_figure
from pathlib import Path
import svgutils

dpi = 600
save_at = Path("./../../source/_static/")
fmts = ["png", "jpg"]
save_axes(ax[1], name="icon", save_at=save_at, save_fmts=fmts, dpi=dpi)

ax[0].set_title("Synthetic Data")
ax[1].set_title("Covariance Ellipses and PCA Vectors")
ax[-2].set_title("Individual Density, with Contours")
ax[-1].set_title("Overall Density")

save_axes(
    ax[0],
    name="logo_eg_points",
    save_at=save_at,
    save_fmts=fmts,
    dpi=dpi,
    pad=[0, -0.01],
)
save_axes(
    ax[1],
    name="logo_eg_ellipses",
    save_at=save_at,
    save_fmts=fmts,
    dpi=dpi,
    pad=[0, -0.01],
)
save_axes(
    ax[2],
    name="logo_eg_contours",
    save_at=save_at,
    save_fmts=fmts,
    dpi=dpi,
)
save_axes(
    ax[3],
    name="logo_eg_density",
    save_at=save_at,
    save_fmts=fmts,
    dpi=dpi,
)
plt.tight_layout()

save_figure(fig, name="logo_eg_all", save_at=save_at, save_fmts=fmts, dpi=dpi)

"""
svg = svgutils.transform.fromfile(save_at / "logo_eg_points.svg")
originalSVG = svgutils.compose.SVG(save_at / "logo_eg_points.svg")
originalSVG.rotate(180)
figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
figure.save(save_at / "logo_eg_points_180.svg")
"""
