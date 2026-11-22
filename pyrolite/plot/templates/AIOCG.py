import matplotlib

from ...util.plot.axes import init_axes
from ...util.classification import AIOCG as AIOCGClassifier
from ...util.meta import sphinx_doi_link, update_docstring_references, subkwargs
from ...util.log import Handle

logger = Handle(__name__)


@update_docstring_references
def AIOCG(
    ax: matplotlib.axes.Axes | None = None,
    add_labels: bool = False,
    relim: bool = True,
    color: str = "k",
    add_minerals: bool = True,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
     IOCG alteration discrimination diagram for main IOCG alteration types (Na, Ca-Fe, K-Fe and K)
     from Montreuil et al. (2013) [#ref_1]_ to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add the diagram to.
    add_labels : bool
        Whether to add labels at polygon centroids.
    relim : bool
        Whether to relimit axes to the default range for this diagram.
    color : str
        Color for the polygon edges in the diagram.
    add_minerals : bool
        Whether to add stochastic reference minerals to the diagram.

    References
    -----------
    .. [#ref_1] Montreuil J F, Corriveau L, and Grunsky E C (2013). Compositional data analysis of hydrothermal
    alteration in IOCG systems, Great Bear magmatic zone, Canada: to each alteration type its own geochemical signature.
            Geochemistry: Exploration, Environment, Analysis, 13 (4):229-247.
            doi:{Montreuil2013}
    """
    ax = init_axes(ax=ax, **kwargs)

    clf = AIOCGClassifier()

    ax = clf.add_to_axes(
        ax=ax, color=color, add_labels=add_labels, **kwargs
    )

    if relim:
        ax.set(xlim=clf.lims["xlim"], ylim=clf.lims["ylim"])
    return ax



AIOCG.__doc__ = AIOCG.__doc__.format(
    Montreuil2013=sphinx_doi_link("10.1144/geochem2011-101")
)
