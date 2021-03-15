import matplotlib.pyplot as plt
import numpy as np
from ...util.plot.axes import init_axes
from ...util.classification import TAS as TASclassifier
from ...util.meta import sphinx_doi_link, update_docstring_references, subkwargs
from ...util.log import Handle

logger = Handle(__name__)


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
    TAS_xlim, TAS_ylim = (30, 90), (0, 20)
    if ax is None:
        xlim, ylim = TAS_xlim, TAS_ylim
    else:
        # if the axes limits are not defaults, update to reflect the axes
        ax_defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, TAS_xlim][np.allclose(ax_xlim, ax_defaults)],
            [ax_ylim, TAS_ylim][np.allclose(ax_ylim, ax_defaults)],
        )
    ax = init_axes(ax=ax, **kwargs)

    tas = TASclassifier()
    tas.add_to_axes(ax=ax, **kwargs)
    if relim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


TAS.__doc__ = TAS.__doc__.format(LeBas1992=sphinx_doi_link("10.1007/BF01160698"))
