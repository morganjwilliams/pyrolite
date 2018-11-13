import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from pyrolite.geochem import REE, get_radii
from pyrolite.util.math import lambdas, lambda_poly_func, OP_constants
#from pyrolite.util.pd import *
#from pyrolite.util.text import titlecase


def plot_lambda_decomp(ax, xs, weights, params, log=False, **kwargs):
    """Plot polynomials on an axis over x values."""
    for w, p in zip(weights, params): # plot the polynomials
        f = np.ones_like(xs) * w
        for c in p:
            f *= (xs - np.float(c))
        if log:
            f = np.exp(f)
        ax.plot(xs, f, label='$x^{}$'.format(len(p)), **kwargs)


def REE_plot(ax=None, exclude=[]):
    """Creates standard a REE diagram."""
    if ax is not None:
        fig = ax.figure
        ax = ax
    else:
        fig, ax = plt.subplots()
    ree = [i for i in REE() if not i in exclude]
    radii = np.array(get_radii(ree))

    _ax = ax.twiny()
    ax.set_yscale('log')
    ax.set_xlim((0.99 * radii.min(), 1.01 * radii.max()))
    _ax.set_xticks(radii)
    _ax.set_xticklabels(ree)
    _ax.set_xlim(ax.get_xlim())
    _ax.set_xlabel('Element')
    ax.axhline(1., ls='--', c='k', lw=0.5)
    ax.set_ylabel(' $\mathrm{X / X_{Reference}}$')
    ax.set_xlabel('Ionic Radius ($\mathrm{\AA}$)')

    return ax


ree = [i for i in REE() if not i in ['Pm', 'Eu']]
radii = np.array(get_radii(ree))

# Some fake log(X) data
lnY = np.random.randn(*radii.shape)*0.2 + \
      np.linspace(3., 0., radii.size) + \
      (radii - 1.11)**2. -0.1

Y = np.exp(lnY)

ax = REE_plot()
ax.plot(radii, Y, label='Example Data')
ax.scatter(radii, Y)
params = OP_constants(radii, degree=4)

ls = lambdas(np.log(Y), radii, params=params, degree=4)
ls
_radii = np.linspace(radii[0], radii[-1], 20)
plot_lambda_decomp(ax, _radii, ls, params, log=True,
                   alpha=0.5)

l_func = lambda_poly_func(ls, pxs=radii, params=params)

ax.plot(_radii, np.exp(l_func(_radii)),
        label='Reconstructed\nLambdas',
        c='k',)
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))


np.random.rand(4)

fig, ax = plt.subplots(2)

ls = np.random.rand(4)


l_func = lambda_poly_func(ls, pxs=radii, params=params)

ax[0].plot(_radii, np.exp(l_func(_radii)[0]), label='Reconstructed Lambdas')

ax[2].set_xlabel('λ$_0$')
ax[2].set_ylabel('λ$_1$')

ax[0].set_ylabel('{} / {}'.format(r2, ref))
ax[0].set_ylim((0, 3))
ax[0].legend(frameon=False, loc='lower left')
ax[1].legend(frameon=False, ncol=2, loc='lower center')
ax[1].set_ylabel('Component Weighted Value')

for a in ax[:2]:
    a.set_xlabel('Ionic Radius')

plt.tight_layout()
