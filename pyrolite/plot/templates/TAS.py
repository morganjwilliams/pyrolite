import matplotlib.pyplot as plt
import numpy as np
from ...util.classification import Geochemistry
from ...util.meta import sphinx_doi_link, update_docstring_references, subkwargs


@update_docstring_references
def TAS(ax=None, relim=True, color="k", **kwargs):
    """
    Adds the TAS diagram from Le Bas (1992) [#ref_1]_ to an axes.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.
    relim : :class:`bool`
        Whether to relimit axes to fit the built in ranges for this diagram.
    color : :class:`str`
        Line color for the diagram.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`

    References
    -----------
    .. [#ref_1] Le Bas, M.J., Le Maitre, R.W., Woolley, A.R., 1992.
                The construction of the Total Alkali-Silica chemical
                classification of volcanic rocks.
                Mineralogy and Petrology 46, 1–22.
                doi: {LeBas1992}

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


for f in [TAS]:
    f.__doc__ = f.__doc__.format(LeBas1992=sphinx_doi_link("10.1007/BF01160698"))
