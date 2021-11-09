import matplotlib.pyplot as plt
import numpy as np

from ...util.classification import QAP as QAPclassifer
from ...util.log import Handle
from ...util.meta import (sphinx_doi_link, subkwargs,
                          update_docstring_references)
from ...util.plot.axes import init_axes

logger = Handle(__name__)


@update_docstring_references
def QAP(ax=None, add_labels=False, which_labels="ID", color="k", **kwargs):
    """
    IUGS QAP ternary classification diagram [#ref_1]_ [#ref_2]_.

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
    .. [#ref_1] Streckeisen, A. Classification and nomenclature of plutonic rocks
                recommendations of the IUGS subcommission on the systematics of
                Igneous Rocks. Geol Rundsch 63, 773–786 (1974).
                doi: {Streckeisen1974}
    .. [#ref_2] Le Maitre,R.W. 2002. Igneous Rocks: A Classification and Glossary
                of Terms : Recommendations of International Union of Geological
                Sciences Subcommission on the Systematics of Igneous Rocks.
                Cambridge University Press, 236pp
    """
    clf = QAPclassifer()
    clf.add_to_axes(
        ax=ax, color=color, add_labels=add_labels, which_labels=which_labels, **kwargs
    )
    return ax


QAP.__doc__ = QAP.__doc__.format(Streckeisen1974=sphinx_doi_link("10.1007/BF01820841"))
