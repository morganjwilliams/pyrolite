"""
Line interpolation for matplotlib lines and paths.
"""

import matplotlib.collections
import matplotlib.path
import numpy as np
import scipy.interpolate

from ..log import Handle

logger = Handle(__name__)


def interpolate_path(
    path, resolution=100, periodic=False, aspath=True, closefirst=False, **kwargs
):
    """
    Obtain the interpolation of an existing path at a given
    resolution. Keyword arguments are forwarded to
    :func:`scipy.interpolate.splprep`.

    Parameters
    -----------
    path : :class:`matplotlib.path.Path`
        Path to interpolate.
    resolution :class:`int`
        Resolution at which to obtain the new path. The verticies of
        the new path will have shape (`resolution`, 2).
    periodic : :class:`bool`
        Whether to use a periodic spline.
    periodic : :class:`bool`
        Whether to return a :code:`matplotlib.path.Path`, or simply
        a tuple of x-y arrays.
    closefirst : :class:`bool`
        Whether to first close the path by appending the first point again.

    Returns
    --------
    :class:`matplotlib.path.Path` | :class:`tuple`
        Interpolated :class:`~matplotlib.path.Path` object, if
        `aspath` is :code:`True`, else a tuple of x-y arrays.
    """
    x, y = path.vertices.T
    if x.size > 4:
        if closefirst:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        # s=0 forces the interpolation to go through every point

        tck, _ = scipy.interpolate.splprep(
            [x[:-1], y[:-1]], s=0, per=periodic, **kwargs
        )
        xi, yi = scipy.interpolate.splev(np.linspace(0.0, 1.0, resolution), tck)
        # could get control points for path and construct codes here
        codes = None
        pth = matplotlib.path.Path(np.vstack([xi, yi]).T, codes=codes)
        if aspath:
            return pth
        else:
            return pth.vertices.T
    else:
        return path.vertices.T


def interpolated_patch_path(patch, resolution=100, **kwargs):
    """
    Obtain the periodic interpolation of the existing path of a patch at a
    given resolution.

    Parameters
    -----------
    patch : :class:`matplotlib.patches.Patch`
        Patch to obtain the original path from.
    resolution :class:`int`
        Resolution at which to obtain the new path. The verticies of the new path
        will have shape (`resolution`, 2).

    Returns
    --------
    :class:`matplotlib.path.Path`
        Interpolated :class:`~matplotlib.path.Path` object.
    """
    pth = patch.get_path()
    tfm = patch.get_transform()
    pathtfm = tfm.transform_path(pth)
    return interpolate_path(
        pathtfm, resolution=resolution, aspath=True, periodic=True, **kwargs
    )


def get_contour_paths(src, resolution=100, minsize=3, filter=True):
    """
    Extract the paths of contours from a contour plot.

    Parameters
    ------------
    ax : :class:`matplotlib.axes.Axes` | `matplotlib.contour.QuadContourSet`
        Axes to extract contours from.
    resolution : :class:`int`
        Resolution of interpolated splines to return.
    filter : bool
        Whether to filter out paths which have no length.

    Returns
    --------
    contourspaths : :class:`list` (:class:`list`)
        List of lists, each represnting one line collection (a single contour). In the
        case where this contour is multimodal, there will be multiple paths for each
        contour.
    contournames : :class:`list`
        List of names for contours, where they have been labelled, and there are no
        other text artists on the figure.
    contourstyles : :class:`list`
        List of styles for contours.

    """
    if isinstance(src, matplotlib.axes.Axes):

        def _iscontour(c):
            # contours/default lines don't have markers - allows distinguishing scatter
            return isinstance(c, matplotlib.contour.QuadContourSet)

        collections = [
            c
            for c in src.collections
            if (_iscontour(c) and (len(c.get_paths()) if filter else True))
        ]
        if len(collections) == 1:
            src = collections[0]
        else:
            raise NotImplementedError("Multiple contour sets found on axes.")
    elif isinstance(src, matplotlib.contour.ContourSet):
        pass
    names = src.levels
    paths = [c for c in src._paths]  #

    interp_paths = [
        interpolate_path(
            p,
            resolution=resolution,
            periodic=True,
            aspath=False,
        )
        for p in paths
    ]
    edgecolors = [{"color": c} for c in src.get_edgecolor()]
    names
    if not filter:
        return interp_paths, names, edgecolors
    else:
        return (
            [p for p in interp_paths if p.size],
            [n for p, n in zip(interp_paths, names) if p.size],
            [c for p, c in zip(interp_paths, edgecolors) if p.size],
        )
