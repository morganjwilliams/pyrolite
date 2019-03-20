import numpy as np
import pandas as pd
from collections import defaultdict
import ternary as pyternary
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import matplotlib.path
import matplotlib.cm
from pyrolite.plot import density
from pyrolite.comp.codata import *
from pyrolite.util.plot import *
from pyrolite.util.skl import *
from pyrolite.util.math import eigsorted
from sklearn.decomposition import PCA

# %% -- colors
np.random.seed(82)
tableau10blind = [
    (0, 107, 164),
    # (255, 128, 14),
    (171, 171, 171),
    (89, 89, 89),
    (95, 158, 209),
    (200, 82, 0),
    (137, 137, 137),
    (163, 200, 236),
    (255, 188, 121),
    (207, 207, 207),
]
for i in range(len(tableau10blind)):
    r, g, b = tableau10blind[i]
    tableau10blind[i] = (r / 255.0, g / 255.0, b / 255.0)

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=tableau10blind)

# %% --
covs = np.array(
    [
        [[1, 0], [0, 1]],
        [[0.5, 0.15], [0.15, 0.5]],
        [[1.5, -1], [-1, 1.5]],
        [[1.2, -0.6], [-0.6, 1.2]],
    ]
)
# %%  --
scale = 100
fig, axes = plt.subplots(1, 3, figsize=(15, 5 * np.sqrt(3) / 2))
ax, ax1, ax2 = axes
f, tax0 = pyternary.figure(ax=ax, scale=scale)
f1, tax1 = pyternary.figure(ax=ax1, scale=scale)
f2, tax2 = pyternary.figure(ax=ax2, scale=scale)
ax.tax, ax1.tax, ax2.tax = tax0, tax1, tax2

tpt_collections = [] # container for point collections
dvars = ["A", "B", "C"]
T = ALRTransform()
pca = PCA(n_components=2)
dist = 1.0
sig = 0.1
for ix, (cx, cy, radius, C) in enumerate(
    [
        (-dist, -dist, sig, covs[0]),
        (dist, -dist, sig, covs[1]),
        (-dist, dist, sig, covs[2]),
        (dist, dist, sig, covs[3]),
    ]
):
    mean = [cx, cy]
    cov = C * radius
    color = tableau10blind[ix]
    x, y = np.random.multivariate_normal(mean, cov, 2000).T  # random in R
    points = np.vstack((x, y)).T  # points in logspace
    txyz = T.inverse_transform(points) * scale
    tpt_collections.append(txyz)
    ax.tax.scatter(txyz, marker="D", alpha=0.3, s=5)
    # ellipses
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1]))
    for nstd in np.arange(1, 4)[::-1]:  # backwards for svg construction
        xsig, ysig = 2 * nstd * np.sqrt(vals)
        ell = matplotlib.patches.Ellipse(
            xy=(np.mean(x), np.mean(y)), width=xsig, height=ysig, angle=theta[:1]
        )
        path = interpolated_patch_path(ell, resolution=100)
        tpts = (
            T.inverse_transform(path.vertices) * 100
        )  # transform to compositional data
        xy = ABC_to_xy(tpts, yscale=np.sqrt(3) / 2)
        # xy[:, 1] *= np.sqrt(3) / 2
        xy *= scale
        patch = matplotlib.patches.PathPatch(matplotlib.path.Path(xy))
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(1 / nstd)
        patch.set_linewidth(0.5)
        ax1.ax.add_artist(patch)

    # vectors
    pca.fit(points)

    for variance, vector in zip(pca.explained_variance_, pca.components_):
        v1 = pca.mean_
        v2 = pca.mean_ + vector * 2 * np.sqrt(variance)
        v1 = T.inverse_transform(v1[np.newaxis, :])
        v2 = T.inverse_transform(v2[np.newaxis, :])
        v1 = v1 * 100 / v2.sum(axis=1)[0]
        v2 = v2 * 100 / v2.sum(axis=1)[0]
        line = vector_to_line(pca.mean_, vector, variance, spans=3)
        line = T.inverse_transform(line)
        line = line * 100 / line.sum(axis=1)[0]
        xy = ABC_to_xy(line, yscale=np.sqrt(3) / 2)
        xy *= scale
        ax.plot(*xy.T, ls="-", lw=0.5, color="k")


txyz = np.vstack(tpt_collections)
xe, ye, zi = ternary_heatmap(txyz, bins=100, mode="density", aspect="eq")
zi[np.isnan(zi)] = 0.0
ls, vs = percentile_contour_values_from_meshz(zi, percentiles=[0.95, 0.05])
norm = matplotlib.colors.Normalize(vmin=vs[0], vmax=np.nanmax(zi))
xi, yi = bin_edges_to_centres(xe), bin_edges_to_centres(ye)

plot_Z_percentiles(
    xi,
    yi,
    zi.T,
    percentiles=[0.5],
    ax=ax2,
    cmap="Blues_r",
    norm=norm,
    label_contours=False,
)

cmap = matplotlib.cm.Greys
cmap.set_under(color=(1, 1, 1, 0.0))  # transparent white
ax2.pcolormesh(xi, yi, zi, cmap=cmap, norm=norm)


for a in [ax, ax1, ax2]:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.spines["left"].set_visible(False)
    a.get_xaxis().set_ticks([])
    a.get_yaxis().set_ticks([])
    a.patch.set_facecolor(None)
    a.patch.set_visible(False)
    a.set_aspect("equal")
    a.tax.boundary(linewidth=1.0)
plt.tight_layout()

# %% Save Figure
from pyrolite.util.plot import save_axes
from pathlib import Path
import svgutils

dpi = 900

save_at = Path("./../../../source/_static/")
fmts = ["svg", "png", "jpg"]
save_axes(ax, name="icon", save_at=save_at, save_fmts=fmts, dpi=dpi)
save_axes(ax1, name="icon_points", save_at=save_at, save_fmts=fmts, dpi=dpi)
save_axes(ax2, name="icon_density", save_at=save_at, save_fmts=fmts, dpi=dpi)

svg = svgutils.transform.fromfile("icon_points.svg")
originalSVG = svgutils.compose.SVG("icon_points.svg")
originalSVG.rotate(180)
figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
figure.save("icon_points_180.svg")


svg = svgutils.transform.fromfile("icon_density.svg")
originalSVG = svgutils.compose.SVG("icon_density.svg")
originalSVG.rotate(180)
figure = svgutils.compose.Figure(svg.height, svg.width, originalSVG)
figure.save("icon_density_180.svg")
