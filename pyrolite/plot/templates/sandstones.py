import numpy as np

from ...util.classification import Herron as Herronclassifier
from ...util.classification import Pettijohn as PJclassifier
from ...util.log import Handle
from ...util.meta import sphinx_doi_link, update_docstring_references
from ...util.plot.axes import init_axes

logger = Handle(__name__)


@update_docstring_references
def Pettijohn(
    ax=None, add_labels=False, which_labels="ID", relim=True, color="k", **kwargs
):
    """
    Adds the Pettijohn (1973) [#ref_1] sandstones classification diagram.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.
    add_labels : :class:`bool`
        Whether to add labels at polygon centroids.
    which_labels : :class:`str`
        Which data to use for field labels - field 'name' or 'ID'.
    relim : :class:`bool`
        Whether to relimit axes to fit the built in ranges for this diagram.
    color : :class:`str`
        Line color for the diagram.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`

    References
    -----------
    .. [#ref_1] Pettijohn, F. J., Potter, P. E. and Siever, R. (1973).
                Sand  and Sandstone. New York, Springer-Verlag. 618p.
                doi: {Pettijohn1973}

    """
    PJ_xlim, PJ_ylim = (0, 2.5), (-1.5, 1)
    if ax is None:
        xlim, ylim = PJ_xlim, PJ_ylim
    else:
        # if the axes limits are not defaults, update to reflect the axes
        ax_defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, PJ_xlim][np.allclose(ax_xlim, ax_defaults)],
            [ax_ylim, PJ_ylim][np.allclose(ax_ylim, ax_defaults)],
        )
    ax = init_axes(ax=ax, **kwargs)

    pjc = PJclassifier()
    pjc.add_to_axes(ax=ax, add_labels=add_labels, which_labels=which_labels, **kwargs)
    if relim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


@update_docstring_references
def Herron(
    ax=None, add_labels=False, which_labels="ID", relim=True, color="k", **kwargs
):
    """
    Adds the Herron (1988) [#ref_1] sandstones classification diagram.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.
    add_labels : :class:`bool`
        Whether to add labels at polygon centroids.
    which_labels : :class:`str`
        Which data to use for field labels - field 'name' or 'ID'.
    relim : :class:`bool`
        Whether to relimit axes to fit the built in ranges for this diagram.
    color : :class:`str`
        Line color for the diagram.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`

    References
    -----------
    .. [#ref_1] Herron, M.M. (1988).
                Geochemical classification of terrigenous sands and shales
                from core or log data.
                Journal of Sedimentary Research, 58(5), pp.820-829.
                doi: {Herron1988}

    """
    Herron_xlim, Herron_ylim = (0, 2.5), (-1.5, 2)
    if ax is None:
        xlim, ylim = Herron_xlim, Herron_ylim
    else:
        # if the axes limits are not defaults, update to reflect the axes
        ax_defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, Herron_xlim][np.allclose(ax_xlim, ax_defaults)],
            [ax_ylim, Herron_ylim][np.allclose(ax_ylim, ax_defaults)],
        )
    ax = init_axes(ax=ax, **kwargs)

    hc = Herronclassifier()
    hc.add_to_axes(ax=ax, add_labels=add_labels, which_labels=which_labels, **kwargs)
    if relim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


Pettijohn.__doc__ = Pettijohn.__doc__.format(
    Pettijohn1973=sphinx_doi_link("10.1007/978-1-4615-9974-6"),
)
Herron.__doc__ = Herron.__doc__.format(
    Herron1988=sphinx_doi_link("10.1306/212F8E77-2B24-11D7-8648000102C1865D"),
)
