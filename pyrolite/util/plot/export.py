"""
Functions for export of figures and figure elements from matplolib.
"""
import os
import numpy as np
import matplotlib.path
import matplotlib.transforms
from ..log import Handle

logger = Handle(__name__)


def save_figure(
    figure, name="fig", save_at="", save_fmts=["png"], output=False, **kwargs
):
    """
    Save a figure at a specified location in a number of formats.
    """
    default_config = dict(bbox_inches="tight", transparent=True)
    config = default_config.copy()
    config.update(kwargs)
    for fmt in save_fmts:
        out_filename = os.path.join(str(save_at), name + "." + fmt)
        if output:
            logger.info("Saving " + out_filename)
        figure.savefig(out_filename, format=fmt, **config)


def save_axes(ax, name="fig", save_at="", save_fmts=["png"], pad=0.0, **kwargs):
    """
    Save either a single or multiple axes (from a single figure) based on their
    extent. Uses the save_figure procedure to save at a specific location using
    a number of formats.

    Todo
    -----
        * Add legend to items
    """
    # Check if axes is a single axis or list of axes

    if isinstance(ax, matplotlib.axes.Axes):
        extent = get_full_extent(ax, pad=pad)
        figure = ax.figure
    else:
        extent_items = []
        for a in ax:
            extent_items.append(get_full_extent(a, pad=pad))
        figure = ax[0].figure
        extent = matplotlib.transforms.Bbox.union([item for item in extent_items])
    save_figure(
        figure,
        bbox_inches=extent,
        save_at=save_at,
        name=name,
        save_fmts=save_fmts,
        **kwargs
    )


def get_full_extent(ax, pad=0.0):
    """
    Get the full extent of an axes, including axes labels, tick labels, and
    titles. Text objects are first drawn to define the extents.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes of which to check items to get full extent.
    pad : :class:`float` | :class:`tuple`
        Amount of padding to add to the full extent prior to returning. If a tuple is
        passed, the padding will be as above, but for x and y directions, respectively.

    Returns
    -------
    :class:`matplotlib.transforms.Bbox`
        Bbox of the axes with optional additional padding.

    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.renderer

    items = [ax]

    if len(ax.get_title()):
        items += [ax.title]

    for a in [ax.xaxis, ax.yaxis]:
        if len(a.get_label_text()):
            items += [a.label]

    for t_lb in [ax.get_xticklabels(), ax.get_yticklabels()]:
        if np.array([len(i.get_text()) > 0 for i in t_lb]).any():
            items += t_lb

    bbox = matplotlib.transforms.Bbox.union(
        [item.get_window_extent(renderer) for item in items]
    )
    if isinstance(pad, (float, int)):
        full_extent = bbox.expanded(1.0 + pad, 1.0 + pad)
    elif isinstance(pad, (list, tuple)):
        full_extent = bbox.expanded(1.0 + pad[0], 1.0 + pad[1])
    else:
        raise NotImplementedError
    return full_extent.transformed(ax.figure.dpi_scale_trans.inverted())


def path_to_csv(path, xname="x", yname="y", delim=", ", linesep=os.linesep):
    """
    Extract the verticies from a path and write them to csv.

    Parameters
    ------------
    path : :class:`matplotlib.path.Path` | :class:`tuple`
        Path or x-y tuple to use for coordinates.
    xname : :class:`str`
        Name of the x variable.
    yname : :class:`str`
        Name of the y variable.
    delim : :class:`str`
        Delimiter for the csv file.
    linesep : :class:`str`
        Line separator character.

    Returns
    -------
    :class:`str`
        String-representation of csv file, ready to be written to disk.
    """
    if isinstance(path, matplotlib.path.Path):
        x, y = path.vertices.T
    else:  # isinstance(path, (tuple, list))
        x, y = path

    header = [xname, yname]
    datalines = [[x, y] for (x, y) in zip(x, y)]
    content = linesep.join(
        [delim.join(map(str, line)) for line in [header] + datalines]
    )
    return content
