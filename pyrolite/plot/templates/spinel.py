import matplotlib.pyplot as plt
import numpy as np

from ...util.classification import SpinelFeBivariate as SpinelBivariate
from ...util.classification import SpinelTrivalentTernary as SpinelTrivalent
from ...util.log import Handle
from ...util.meta import sphinx_doi_link, subkwargs, update_docstring_references
from ...util.plot.axes import init_axes

logger = Handle(__name__)


def SpinelFeBivariate(
    ax=None, add_labels=False, which_labels="ID", color="k", **kwargs
):
    """
    Fe-Spinel classification, designed for data in atoms per formula unit.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the diagram to.
    add_labels : :class:`bool`
        Whether to add labels at polygon centroids.
    which_labels : :class:`str`
        Which data to use for field labels - field 'name' or 'ID'.
    color : :class:`str`
        Color for the polygon edges in the diagram.
    """
    ax = init_axes(ax=ax, **kwargs)

    clf = SpinelBivariate()
    clf.add_to_axes(
        ax=ax,
        color=color,
        add_labels=add_labels,
        which_labels=which_labels,
        **kwargs,
    )
    return ax


def SpinelTrivalentTernary(
    ax=None, add_labels=False, which_labels="ID", color="k", **kwargs
):
    """
    Spinel Trivalent Ternary classification  - designed for data in atoms per
    formula unit.

    Parameters
    -----------
    ax : :class:`matplotlib.axes.Axes`
        Ternary axes to add the diagram to.
    add_labels : :class:`bool`
        Whether to add labels at polygon centroids.
    which_labels : :class:`str`
        Which data to use for field labels - field 'name' or 'ID'.
    color : :class:`str`
        Color for the polygon edges in the diagram.
    """
    ax = init_axes(ax=ax, projection="ternary", **kwargs)

    clf = SpinelTrivalent()
    clf.add_to_axes(
        ax=ax,
        color=color,
        add_labels=add_labels,
        which_labels=which_labels,
        **kwargs,
    )
    return ax
