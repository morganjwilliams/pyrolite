import matplotlib.pyplot as plt
import numpy as np

from ...util.classification import USDASoilTexture as USDAclassifier
from ...util.log import Handle
from ...util.meta import (sphinx_doi_link, subkwargs, update_docstring_references)
from ...util.plot.axes import init_axes

logger = Handle(__name__)


@update_docstring_references
def Jensen_Plot(ax=None, add_labels=False, color="k", **kwargs):
    """
    Jensen Plot for classification of sub-alkaline volcanic rocks 
    [#ref_1]_.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.
    add_labels : :class:`bool`
        Whether to include the labels for the diagram.
    color : :class:`str`
        Line color for the diagram.

    # Returns
    # -------
    # ax : :class:`matplotlib.axes.Axes`

    # References
    # -----------
    # .. [#ref_1] Jensen, L. S. (1976) A new cation plot for classifying sub-alkaline volcanic rocks. 
    #             Ontario Division Mines. Miscellaneous Paper No. 66.
    # """
    ax = init_axes(ax=ax, projection="ternary", **kwargs)

    clf = JensenPlot()
    clf.add_to_axes(ax=ax, color=color, add_labels=add_labels, **kwargs)
    return ax



