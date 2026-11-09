import matplotlib.axes

from ...util.classification import Magnetite as MagnetiteClassifier
from ...util.log import Handle
from ...util.meta import sphinx_doi_link, update_docstring_references
from ...util.plot.axes import init_axes

logger = Handle(__name__)


@update_docstring_references
def Magnetite(
    ax: matplotlib.axes.Axes | None = None,
    add_labels: bool = False,
    which_labels: str = "ID",
    which_model: str | None = None,
    relim: bool = True,
    color: str = "k",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Magnetite trace element discrimination diagram for mineral deposit types
    from Dupuis and Beaudoin (2011) [#ref_1]_.

    Parameters
    -----------
    ax : matplotlib.axes.Axes
        Axes to add the diagram to.
    add_labels : bool
        Whether to add labels at polygon centroids.
    which_labels : str
        Which data to use for field labels - field 'name' or 'ID'.
    which_model : str
        Diagram to use: 'TiV_CaAlMn' (Ca+Al+Mn vs Ti+V; default) or
        'TiV_NiCrMn' (Ni/(Cr+Mn) vs Ti+V).
    relim : bool
        Whether to relimit axes to the default range for this diagram.
    color : str
        Color for the polygon edges in the diagram.

    References
    -----------
    .. [#ref_1] Dupuis, C., Beaudoin, G. (2011).
                Discriminant diagrams for iron oxide trace element fingerprinting
                of mineral deposit types. Mineralium Deposita 46, 319-335.
                doi: {DupuisBeaudoin2011}
    .. [#ref_2] Nadoll, P. (2011).
                Geochemistry of magnetite from hydrothermal ore deposits and host rocks –
                Case studies from the Proterozoic Belt Supergroup, Cu-Mo-porphyry + skarn
                and Climax-Mo deposits in the western United States.
                doi: {Nadoll2011}
    .. [#ref_3] Nadoll, P., Angerer, T., Mauk, J. L., French, D., & Walshe, J. (2014).
                The chemistry of hydrothermal magnetite: A review. Ore Geology Reviews, 61, 1–32.
                doi: {Nadoll2014}
    """
    ax = init_axes(ax=ax, **kwargs)

    clf = MagnetiteClassifier(which_model=which_model)
    ax = clf.add_to_axes(
        ax=ax, color=color, add_labels=add_labels, which_labels=which_labels, **kwargs
    )
    if relim:
        ax.set(xlim=clf.lims["xlim"], ylim=clf.lims["ylim"])
    return ax


Magnetite.__doc__ = Magnetite.__doc__.format(
    DupuisBeaudoin2011=sphinx_doi_link("10.1007/s00126-011-0334-y"),
    Nadoll2011=sphinx_doi_link("https://hdl.handle.net/2292/7512"),
    Nadoll2014=sphinx_doi_link("10.1016/j.oregeorev.2013.12.013")
)