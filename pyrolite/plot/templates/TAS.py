import matplotlib.pyplot as plt
from ...util.classification import Geochemistry
from ...util.meta import subkwargs

# @update_docstring_references
def TAS(ax=None, relim=True, color="k", **kwargs):
    """
    Adds the TAS diagram to an axes.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
    """
    xlim, ylim = (30, 90), (0, 20)
    if ax is None:
        fig, ax = plt.subplots(1, **subkwargs(kwargs, plt.subplots, plt.figure))
    else:
        # if the axes limits are not defaults, update to reflect the axes
        defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, xlim][np.allclose(ax_xlim, defaults)],
            [ax_ylim, ylim][np.allclose(ax_ylim, defaults)],
        )
    tas = Geochemistry.TAS()
    tas.add_to_axes(ax=ax, **kwargs)
    if relim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax
