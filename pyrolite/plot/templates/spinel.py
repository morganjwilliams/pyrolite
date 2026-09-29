import matplotlib.axes

from ...util.classification import SpinelFeBivariate as SpinelBivariate
from ...util.classification import SpinelTrivalentTernary as SpinelTrivalent
from ...util.log import Handle
from ...util.plot.axes import init_axes

logger = Handle(__name__)


def SpinelFeBivariate(
    ax: matplotlib.axes.Axes | None = None,
    add_labels: bool = False,
    which_labels: str = "ID",
    color: str = "k",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Fe-Spinel classification, designed for data in atoms per formula unit.

    Parameters
    -----------
    ax : matplotlib.axes.Axes
        Axes to add the diagram to.
    add_labels : bool
        Whether to add labels at polygon centroids.
    which_labels : str
        Which data to use for field labels - field 'name' or 'ID'.
    color : str
        Color for the polygon edges in the diagram.
    """
    ax = init_axes(ax=ax, **kwargs)

    clf = SpinelBivariate()
    ax = clf.add_to_axes(
        ax=ax,
        color=color,
        add_labels=add_labels,
        which_labels=which_labels,
        **kwargs,
    )
    return ax


def SpinelTrivalentTernary(
    ax: matplotlib.axes.Axes | None = None,
    add_labels: bool = False,
    which_labels: str = "ID",
    color: str = "k",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Spinel Trivalent Ternary classification  - designed for data in atoms per
    formula unit.

    Parameters
    -----------
    ax : matplotlib.axes.Axes
        Ternary axes to add the diagram to.
    add_labels : bool
        Whether to add labels at polygon centroids.
    which_labels : str
        Which data to use for field labels - field 'name' or 'ID'.
    color : str
        Color for the polygon edges in the diagram.
    """
    ax = init_axes(ax=ax, projection="ternary", **kwargs)

    clf = SpinelTrivalent()
    ax = clf.add_to_axes(
        ax=ax,
        color=color,
        add_labels=add_labels,
        which_labels=which_labels,
        **kwargs,
    )
    return ax
