import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from pandas.plotting import parallel_coordinates
from ..util.meta import subkwargs
from ..util.plot.style import DEFAULT_CONT_COLORMAP
from ..util.plot.axes import init_axes
from ..util.log import Handle

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
    columns=None,
    classes=None,
    rescale=True,
    color_by=None,
    legend=False,
    cmap=DEFAULT_CONT_COLORMAP,
    ax=None,
    alpha=1.0,
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
    columns : :class:`list`
        Subset of dataframe columns to use as indexes along the x-axis.
    rescale : :class:`bool`
        Whether to rescale values to [-1, 1].
    color_by : :class:`str` | :class:`list`
        Column to use as an index for a color value, or a list of color values.
    legend : :class:`bool`, :code:`False`
        Whether to include or suppress the legend.
    cmap : :class:`matplotlib.colors.Colormap`
        Colormap to use for :code:`color_by`.
    ax : :class:`matplotlib.axes.Axes`
        Axis to plot on (optional).
    alpha : :class:`float`
        Coefficient for the alpha channel for the colors, if color_by is specified.

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

    if columns is None:
        columns = samples.columns.tolist()

    color = None
    if color_by is not None:
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if isinstance(color_by, str) and (color_by in columns):  # column name
            cvals = samples[color_by].values
        elif isinstance(color_by, (list, np.ndarray, pd.Series)):  # numeric values.
            cvals = np.array(color_by)
        else:
            raise NotImplementedError(
                "'color_by' must be a column name or 1D array of values."
            )
        norm = matplotlib.colors.Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))
        color = cmap(norm(cvals))
        color[:, -1] *= alpha

    non_target = [i for i in columns if i != target]
    if rescale:
        samples[non_target] = samples.loc[:, non_target].apply(
            lambda x: (x - x.mean()) / x.std()
        )
    parallel_coordinates(
        samples.loc[:, [target] + non_target],
        target,
        ax=ax,
        color=color,
        **subkwargs(kwargs, parallel_coordinates),
    )
    ax.spines["bottom"].set_color("none")
    ax.spines["top"].set_color("none")
    if not legend:
        ax.get_legend().remove()

    if label_rotate is not None:
        [i.set_rotation(label_rotate) for i in ax.get_xticklabels()]

    return ax
