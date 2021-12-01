import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates

from ..util.log import Handle
from ..util.meta import subkwargs
from ..util.plot.axes import init_axes
from ..util.plot.style import DEFAULT_CONT_COLORMAP, linekwargs
from .color import process_color

logger = Handle(__name__)


def plot_across_axes():
    """
    Parallel coordinate plots in matplotlib where each variable has its own axis.

    Notes
    -----
    This stub is a work-in-progress.
    From stackoverflow.com/questions/17543359/drawing-lines-between-two-plots-in-matplotlib
    """
    transFigure = fig.transFigure.inverted()

    coords = [  # get the figure coordinates for the variables
        transFigure.transform(
            ax[i].transData.transform(np.vstack(np.zeros_like(y[i]), y[i]))
        )
        for i in range(ax)
    ]
    coord1 = transFigure.transform(ax1.transData.transform([x[i], y[i]]))
    coord2 = transFigure.transform(ax2.transData.transform([x[i], y[i]]))

    line = matplotlib.lines.Line2D(
        (coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure
    )
    # instead create a collection here
    LineCollection(segs, linewidths=(0.5, 1, 1.5, 2), colors=colors, linestyle="solid")
    # can we add the collection to one axis?
    ax.add_collection(line_segments)


def parallel(
    df,
    components=None,
    classes=None,
    rescale=True,
    legend=False,
    ax=None,
    label_rotate=60,
    **kwargs,
):
    """
    Create a parallel coordinate plot across dataframe columns, with
    individual lines for each row.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe to create a plot from.
    components : :class:`list`
        Subset of dataframe columns to use as indexes along the x-axis.
    rescale : :class:`bool`
        Whether to rescale values to [-1, 1].
    legend : :class:`bool`, :code:`False`
        Whether to include or suppress the legend.
    ax : :class:`matplotlib.axes.Axes`
        Axis to plot on (optional).

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

    [kwargs.pop(x, None) for x in colors.keys()]  # so colors aren't added twice

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
