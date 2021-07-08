import matplotlib.pyplot as plt
import numpy as np
from ...util.plot.axes import init_axes
from ...util.classification import AIOCG as _AIOCG_Classifier
from ...util.meta import sphinx_doi_link, update_docstring_references, subkwargs
from ...util.log import Handle

logger = Handle(__name__)


@update_docstring_references
def AIOCG(ax=None, relim=True, color="k", **kwargs):
    """
    Adds the AIOCG diagram from Montreuil et al. (2013) [#ref_1]_ to an axes.
    NOTE to user:
    x:y are scaled from 1:1 to 600:400 to account for plot dimension
                                                         
    
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
    .. [#ref_1] Montreuil J F, Corriveau L, and Potter E G (2015). Formation of 
            albitite-hosted uranium within IOCG systems: the Southern Breccia, 
            Great Bear magmatic zone, Northwest Territories, Canada. 
            Mineralium Deposita, 50:293-325.
            doi:`<https://doi.org/10.1007/s00126-014-0530-7>`__
    """
    AIOCG_xlim, AIOCG_ylim = (0, 607.4), (0, 405.2)
    if ax is None:
        xlim, ylim = AIOCG_xlim, AIOCG_ylim
    else:
        # if the axes limits are not defaults, update to reflect the axes
        ax_defaults = (0, 1)
        ax_xlim, ax_ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = (
            [ax_xlim, AIOCG_xlim][np.allclose(ax_xlim, ax_defaults)],
            [ax_ylim, AIOCG_ylim][np.allclose(ax_ylim, ax_defaults)],
        )
    ax = init_axes(ax=ax, **kwargs)

    _AIOCG_Classifier().add_to_axes(ax=ax, **kwargs)

    if relim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return ax


AIOCG.__doc__ = AIOCG.__doc__.format(Montreuil2013=sphinx_doi_link("10.1007/s00126-014-0530-7"))
