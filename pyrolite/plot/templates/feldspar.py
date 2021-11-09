import matplotlib.pyplot as plt
import numpy as np

from ...util.classification import FeldsparTernary as Feldspar
from ...util.log import Handle
from ...util.meta import update_docstring_references
from ...util.plot.axes import init_axes

logger = Handle(__name__)


@update_docstring_references
def FeldsparTernary(ax=None, add_labels=False, which_labels="ID", color="k", **kwargs):
    """
    Simplified feldspar classifcation diagram, based on a version printed in the
    second edition of 'An Introduction to the Rock Forming Minerals' (Deer,
    Howie and Zussman).

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

    References
    -----------
    .. [#ref_1] Deer, W. A., Howie, R. A., & Zussman, J. (2013).
        An introduction to the rock-forming minerals (3rd ed.).
        Mineralogical Society of Great Britain and Ireland.
    """
    clf = Feldspar()
    clf.add_to_axes(
        ax=ax, color=color, add_labels=add_labels, which_labels=which_labels, **kwargs
    )
    return ax
