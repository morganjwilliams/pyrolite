import numpy as np
import matplotlib.pyplot as plt
from pyrolite.util.meta import sphinx_doi_link
from ...util.meta import sphinx_doi_link
from .components import *


def pearceThNbYb(ax=None, relim=True, color="k", lw=0.5, **kwargs):
    """
    Adds the Th-Nb-Yb delimiter lines from Pearce (2008) [#ref_1]_ to an axes.
    This configuration uses

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template onto.

    References
    -----------
    .. [#ref_1] Pearce J. A. (2008) Geochemical fingerprinting of oceanic basalts
                with applications to ophiolite classification and the search for
                Archean oceanic crust. Lithos 100, 14–48.
                doi: {pearce2008}

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
    """
    xlim, ylim = (0.1, 100), (0.01, 10)
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        # if the axes limits are not defaults, update to reflect the axes
        defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, xlim][np.allclose(ax_xlim, defaults)],
            [ax_ylim, ylim][np.allclose(ax_ylim, defaults)],
        )

    geom = GeometryCollection(
        Linear2D(slope=12.5, name="Upper Crustal Limit"),
        Linear2D(slope=0.1, name="Upper MORB-OIB Array"),
        Linear2D(slope=0.1 / 3, name="Lower MORB-OIB Array"),
    )
    xs = np.logspace(*np.log([*xlim]), 1000, base=np.e)
    geom.add_to_axes(ax, xs=xs, color=color, lw=lw, **kwargs)

    if relim:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


def pearceTiNbYb(ax=None, relim=True, color="k", linewidth=0.5, annotate=True, **kwargs):
    """
    Adds the Ti-Nb-Yb delimiter lines from Pearce (2008) [#ref_1]_ to an axes.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template onto.

    References
    -----------
    .. [#ref_1] Pearce J. A. (2008) Geochemical fingerprinting of oceanic basalts
                with applications to ophiolite classification and the search for
                Archean oceanic crust. Lithos 100, 14–48.
                doi: {pearce2008}

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
    """

    # Nb/Yb < 1.45 (CI Chondrite) = NMORB, Nb/Yb > 1.45 (CI Chondrite) EMORB
    xlim, ylim = (0.1, 100), (0.1, 10)
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        # if the axes limits are not defaults, update to reflect the axes
        defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, xlim][np.allclose(ax_xlim, defaults)],
            [ax_ylim, ylim][np.allclose(ax_ylim, defaults)],
        )
    xs = np.logspace(*np.log([*xlim]), 1000, base=np.e)
    geom = GeometryCollection(
        LogLinear2D(p0=[0.1, 2], p1=[100, 2.6], name="Upper OIB Limit"),
        LogLinear2D(p0=[0.1, 0.58], p1=[100, 0.75], name="Upper MORB Array"),
        LogLinear2D(p0=[0.1, 0.27], p1=[100, 0.35], name="Lower MORB-OIB Array"),
        Point([0.7, 0.41], c=color, name="NMORB"),
        Point([3.38, 0.41], c=color, name="EMORB"),
        Point([22.35, 1.35], c=color, name="OIB"),
    )
    geom += geom["Lower MORB-OIB Array"].perpendicular_line(
        centre=(1.45, geom["Lower MORB-OIB Array"](1.45)),
        ylim=(geom["Lower MORB-OIB Array"], geom["Upper MORB Array"]),
        name="NMORB - EMORB Divide",
        ls="--",
    )
    geom += LogLinear2D(
        p0=[60, 0.1],
        p1=[10, 3.5],
        ylim=(geom["Upper OIB Limit"], geom["Upper MORB Array"]),
        name="Tholeiitic - Alkalic Divide",
        ls="--",
    )

    geom.add_to_axes(ax, xs=xs, color=color, lw=lw, **kwargs)

    if relim:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


for f in [pearceThNbYb, pearceTiNbYb]:
    f.__doc__ = f.__doc__.format(
        pearce2008=sphinx_doi_link("10.1016/j.lithos.2007.06.016")
    )
