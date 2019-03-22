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
from pyrolite.util.skl import ILRTransform
from pyrolite.util.synthetic import random_composition

np.random.seed(82)
# %% -- colors
t10b3 = [  # tableau 10 colorblind safe colors, a selection of 3
    (r / 255.0, g / 255.0, b / 255.0)
    for r, g, b in [(0, 107, 164), (171, 171, 171), (89, 89, 89), (95, 158, 209)]
]
# %% data and transforms --------------------------------------------------------------
dist = 1.0  # distance from centre
sig = 0.1  # scale for variance
# means for logspace (D=2)
means = np.array([(-dist, -dist), (dist, -dist), (-dist, dist), (dist, dist)])  #
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
T = ILRTransform()
means = T.inverse_transform(means)  # compositional means (D=3)
pts = [random_composition(mean=M, cov=C, size=2000) for M, C in zip(means, covs)]
# %% figure --
scale = 100
fig, ax = plt.subplots(2, 2, figsize=(10, 10 * np.sqrt(3) / 2))
ax = ax.flat
for a in ax:  # append ternary axes
    _, a.tax = pyternary.figure(ax=a, scale=scale)
    a.tax.boundary(linewidth=1.0)

# %% scatter ---------------------------------------------------------------------------
for ix, comp in enumerate(pts):
    color = t10b3[ix]
    ax[0].tax.scatter(comp * scale, marker="D", alpha=0.3, s=5, color=color)
# %% ellipses and vectors from PCA -----------------------------------------------------
for ix, comp in enumerate(pts):
    color = t10b3[ix]
    tfm = T.inverse_transform
    tcomp = T.transform(comp)
    plot_stdev_ellipses(tcomp, ax=ax[1], transform=tfm, nstds=3, color=color)
    plot_pca_vectors(tcomp, ax=ax[1], transform=tfm, nstds=3, ls="-", lw=0.5, color="k")

# %% individual density diagrams ------------------------------------------------------
cmap = matplotlib.cm.Blues
cmap.set_under(color=(1, 1, 1, 0.0))  # transparent white
for ix, comp in enumerate(pts):
    color = t10b3[ix]
    tfm = T.inverse_transform
    xe, ye, zi = ternary_heatmap(comp, bins=100, mode="density", aspect="eq")
    zi[np.isnan(zi)] = 0.0
    l, v = percentile_contour_values_from_meshz(zi, percentiles=[0.95])
    norm = matplotlib.colors.Normalize(vmin=v, vmax=np.nanmax(zi))
    xi, yi = bin_edges_to_centres(xe), bin_edges_to_centres(ye)
    xi, yi = xi * scale, yi * scale

    plot_Z_percentiles(
        xi,
        yi,
        zi,
        percentiles=[0.68, 0.95],
        ax=ax[-2],
        cmap="Blues_r",
        norm=norm,
        labels={0.68: "σ", 0.95: "2σ"},
    )

    ax[-2].pcolor(xi, yi, zi, cmap=cmap, norm=norm)
# %% overall density diagram ----------------------------------------------------------
cmap = matplotlib.cm.Greys
cmap.set_under(color=(1, 1, 1, 0.0))  # transparent white
xe, ye, zi = ternary_heatmap(np.vstack(pts), bins=100, mode="density", aspect="eq")
zi[np.isnan(zi)] = 0.0
ls, vs = percentile_contour_values_from_meshz(zi, percentiles=[0.95, 0.05])
norm = matplotlib.colors.Normalize(vmin=vs[0], vmax=np.nanmax(zi))
xi, yi = bin_edges_to_centres(xe) * scale, bin_edges_to_centres(ye) * scale
ax[-1].pcolormesh(xi, yi, zi, cmap=cmap, norm=norm)
plot_Z_percentiles(
    xi,
    yi,
    zi,
    percentiles=[0.5],
    ax=ax[-1],
    cmap="Blues_r",
    norm=norm,
    label_contours=False,
)
fig
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
from pyrolite.util.plot import save_axes
from pathlib import Path
import svgutils

plt.tight_layout()
dpi = 900

save_at = Path("./../../../source/_static/")
fmts = ["svg", "png", "jpg"]
save_axes(ax[1], name="icon", save_at=save_at, save_fmts=fmts, dpi=dpi)
save_axes(ax[0], name="icon_points", save_at=save_at, save_fmts=fmts, dpi=dpi)
save_axes(ax[2], name="icon_density", save_at=save_at, save_fmts=fmts, dpi=dpi)

svg = svgutils.transform.fromfile(save_at / "icon_points.svg")
originalSVG = svgutils.compose.SVG(save_at / "icon_points.svg")
originalSVG.rotate(180)
figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
figure.save(save_at / "icon_points_180.svg")


svg = svgutils.transform.fromfile(save_at / "icon_density.svg")
originalSVG = svgutils.compose.SVG(save_at / "icon_density.svg")
originalSVG.rotate(180)
figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
figure.save(save_at / "icon_density_180.svg")
