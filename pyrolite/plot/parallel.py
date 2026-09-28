import matplotlib.axes
import pandas as pd
from pandas.plotting import parallel_coordinates

from ..util.log import Handle
from ..util.meta import subkwargs
from ..util.plot.axes import init_axes
from .color import process_color

1
logger = Handle(__name__)


def parallel(
    df: pd.DataFrame,
    components: list[str] | None = None,
    rescale: bool = True,
    legend: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    label_rotate: float = 60,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Create a parallel coordinate plot across dataframe columns, with
    individual lines for each row.

    Parameters
    -----------
    df : pandas.DataFrame
        Dataframe to create a plot from.
    components : list
        Subset of dataframe columns to use as indexes along the x-axis.
    rescale : bool
        Whether to rescale values to [-1, 1].
    legend : bool
        Whether to include or suppress the legend.
    ax : matplotlib.axes.Axes
        Axis to plot on (optional).

    Returns
    -------
    matplotlib.axes.Axes:

    Todo
    ------
    * A multi-axis version would be more compatible with independent rescaling and zoom
    * Enable passing a list of colors
      Rather than just a list of numbers to be converted to colors.
    """
    samples = df.copy()
    ax = init_axes(ax=ax, **kwargs)

    target = samples.index.name or "index"
    samples = samples.reset_index()  # to access the index to use as a 'class'

    if components is None:
        components = samples.columns.tolist()

    non_target = [i for i in components if i != target]
    if rescale:
        samples[non_target] = samples.loc[:, non_target].apply(
            lambda x: (x - x.mean()) / x.std()
        )

    colors = process_color(**kwargs)

    [kwargs.pop(x, None) for x in colors]  # so colors aren't added twice

    parallel_coordinates(
        samples.loc[:, [target] + non_target],
        target,
        ax=ax,
        color=colors.get("color", None),
        **subkwargs(kwargs, parallel_coordinates),
    )
    ax.spines["bottom"].set_color("none")
    ax.spines["top"].set_color("none")
    if not legend:
        ax.get_legend().remove()

    if label_rotate is not None:
        [i.set_rotation(label_rotate) for i in ax.get_xticklabels()]

    return ax
