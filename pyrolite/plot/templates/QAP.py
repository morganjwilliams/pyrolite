import matplotlib.pyplot as plt
import numpy as np
from ...util.plot.axes import init_axes
from ...util.classification import QAP as QAP
from ...util.meta import sphinx_doi_link, update_docstring_references, subkwargs
from ...util.log import Handle

logger = Handle(__name__)


@update_docstring_references
def QAP(ax=None, add_labels=False, color="k", **kwargs):
    """
    IUGS QAP ternary classification
    [#ref_1]_ [#ref_2]_.

    Parameters
    -----------
    name : :class:`str`
        A name for the classifier model.
    axes : :class:`list` | :class:`tuple`
        Names of the axes corresponding to the polygon coordinates.
    fields : :class:`dict`
        Dictionary describing indiviudal polygons, with identifiers as keys and
        dictionaries containing 'name' and 'fields' items.

    References
    -----------
}    .. [#ref_1] Streckeisen, A. Classification and nomenclature of plutonic rocks
                recommendations of the IUGS subcommission on the systematics of
                Igneous Rocks. Geol Rundsch 63, 773–786 (1974).
                doi: {Streckeisen1974}
    .. [#ref_2] Le Maitre,R.W. 2002. Igneous Rocks: A Classification and Glossary
                of Terms : Recommendations of International Union of Geological
                Sciences Subcommission on the Systematics of Igneous Rocks.
                Cambridge University Press, 236pp
    """
    ax = init_axes(ax=ax, **kwargs)
    clf = QAP()
    clf.add_to_axes(ax=ax, color=color, add_labels=add_labels, **kwargs)
    return ax


QAP.__doc__ = QAP.__doc__.format(
    Streckeisen1974=sphinx_doi_link("10.1007/BF01820841")
)
