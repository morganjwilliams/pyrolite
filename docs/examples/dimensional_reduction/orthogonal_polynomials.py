import numpy as np
from pyrolite.plot import REE_radii_plot
from pyrolite.geochem import REE, get_radii
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants

np.random.seed(82)

def plot_orthagonal_polynomial_components(ax, xs, lambdas, params, log=False, **kwargs):
    """Plot polynomials on an axis over x values."""
    for w, p in zip(lambdas, params):  # plot the polynomials
        f = np.ones_like(xs) * w
        for c in p:
            f *= xs - np.float(c)
        if log:
            f = np.exp(f)
        ax.plot(xs, f, label="$x^{}$".format(len(p)), **kwargs)
# %% Generate Some Example Data --------------------------------------------------------
data_ree = [i for i in REE() if not i in ["Pm"]]
data_radii = np.array(get_radii(data_ree))
lnY = (
    np.random.randn(*data_radii.shape) * 0.1
    + np.linspace(3.0, 0.0, data_radii.size)
    + (data_radii - 1.11) ** 2.0
    - 0.1
)

for ix, el in enumerate(data_ree):
    if el in ["Ce", "Eu"]:
        lnY[ix] += np.random.randn(1) * 0.6

Y = np.exp(lnY)
# %% Reduce to Orthogonal Polynomials --------------------------------------------------
exclude = ["Ce", "Eu", "Pm"]
if exclude:
    subset_Y = Y[[i not in exclude for i in data_ree]]
    subset_ree = [i for i in REE() if not i in exclude]
    subset_radii = np.array(get_radii(subset_ree))

params = OP_constants(subset_radii, degree=4)
ls = lambdas(np.log(subset_Y), subset_radii, params=params, degree=4)
continuous_radii = np.linspace(subset_radii[0], subset_radii[-1], 20)
l_func = lambda_poly_func(ls, pxs=subset_radii, params=params)
smooth_profile = np.exp(l_func(continuous_radii))
# %% Plot the Results ------------------------------------------------------------------
ax = REE_radii_plot()
ax.plot(data_radii, Y, marker="D", color='0.5', label="Example Data")
plot_orthagonal_polynomial_components(
    ax, continuous_radii, ls, params, log=True,
)
ax.plot(continuous_radii, smooth_profile, label="Reconstructed\nProfile", c="k")
ax.legend(frameon=False, facecolor=None, bbox_to_anchor=(1, 1))
# %% End -------------------------------------------------------------------------------
from pyrolite.util.plot import save_figure
save_figure(ax.figure, save_at="../../source/_static", name="OrthagPolyDeconstruction")
