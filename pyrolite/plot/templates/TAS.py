import numpy as np

from ...util.classification import TAS as TASclassifier
from ...util.log import Handle
from ...util.meta import sphinx_doi_link, update_docstring_references
from ...util.plot.axes import init_axes

logger = Handle(__name__)


@update_docstring_references
def TAS(
    ax=None,
    add_labels=False,
    which_labels="ID",
    relim=True,
    color="k",
    which_model=None,
    **kwargs,
):
    """
    Adds the TAS diagram to an axes. Diagram from Middlemost (1994) [#ref_1]_,
    a closed-polygon variant after Le Bas et al (1992) [#ref_2]_.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.
    add_labels : :class:`bool`
        Whether to add labels at polygon centroids.
    which_labels : :class:`str`
        Which labels to add to the polygons (e.g. for TAS, 'volcanic', 'intrusive'
        or the field 'ID').
    relim : :class:`bool`
        Whether to relimit axes to fit the built in ranges for this diagram.
    color : :class:`str`
        Line color for the diagram.
    which_model : :class:`str`
        The name of the model variant to use, if not Middlemost.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`

    References
    -----------
    .. [#ref_1] Middlemost, E. A. K. (1994).
                Naming materials in the magma/igneous rock system.
                Earth-Science Reviews, 37(3), 215–224.
                doi: {Middlemost1994}

    .. [#ref_2] Le Bas, M.J., Le Maitre, R.W., Woolley, A.R. (1992).
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

    tas = TASclassifier(which_model=which_model)
    tas.add_to_axes(ax=ax, add_labels=add_labels, which_labels=which_labels, **kwargs)
    if relim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


TAS.__doc__ = TAS.__doc__.format(
    LeBas1992=sphinx_doi_link("10.1007/BF01160698"),
    Middlemost1994=sphinx_doi_link("10.1016/0012-8252(94)90029-9"),
)
