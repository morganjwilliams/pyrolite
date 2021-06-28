import matplotlib.pyplot as plt
import numpy as np
from ...util.plot.axes import init_axes
from ...util.classification import USDASoilTexture as USDAclassifier
from ...util.meta import sphinx_doi_link, update_docstring_references, subkwargs
from ...util.log import Handle

logger = Handle(__name__)


@update_docstring_references
def USDASoilTexture(ax=None, add_labels=False, color="k", **kwargs):
    """
    United States Department of Agriculture Soil Texture classification model
    [#ref_1]_ [#ref_2]_.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to add the template on to.
    add_labels : :class:`bool`
        Whether to include the labels for the diagram.
    color : :class:`str`
        Line color for the diagram.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`

    References
    -----------
    .. [#ref_1] Soil Science Division Staff (2017). Soil survey sand.
                C. Ditzler, K. Scheffe, and H.C. Monger (eds.).
                USDA Handbook 18. Government Printing Office, Washington, D.C.
    .. [#ref_2] Thien, Steve J. (1979). A Flow Diagram for Teaching
                Texture-by-Feel Analysis. Journal of Agronomic Education 8:54–55.
                doi: {Thien1979}
    """
    ax = init_axes(ax=ax, **kwargs)
    clf = USDAclassifier()
    clf.add_to_axes(ax=ax, color=color, add_labels=add_labels, **kwargs)
    return ax


USDASoilTexture.__doc__ = USDASoilTexture.__doc__.format(
    Thien1979=sphinx_doi_link("10.2134/jae.1979.0054")
)
