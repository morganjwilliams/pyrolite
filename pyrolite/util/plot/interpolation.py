"""
Line interpolation for matplotlib lines and paths.
"""

import matplotlib.axes
import matplotlib.contour
import matplotlib.path
import numpy as np
import scipy.interpolate

from ..log import Handle

logger = Handle(__name__)


def interpolate_path(
    path: matplotlib.path.Path,
    resolution: int = 100,
    periodic: bool = False,
    aspath: bool = True,
    closefirst: bool = False,
    **kwargs,
) -> matplotlib.path.Path | np.ndarray:
    """
    Obtain the interpolation of an existing path at a given
    resolution. Keyword arguments are forwarded to
    :func:`scipy.interpolate.splprep`.

    Parameters
    -----------
    path : matplotlib.path.Path
        Path to interpolate.
    resolution int
        Resolution at which to obtain the new path. The verticies of
        the new path will have shape (`resolution`, 2).
    periodic : bool
        Whether to use a periodic spline.
    periodic : bool
        Whether to return a matplotlib.path.Path, or simply
        a tuple of x-y arrays.
    closefirst : bool
        Whether to first close the path by appending the first point again.

    Returns
    --------
    matplotlib.path.Path | tuple
        Interpolated path object, if `aspath` is `True`, else a tuple of x-y arrays.
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


def interpolated_patch_path(
    patch: matplotlib.patches.Patch, resolution: int = 100, **kwargs
) -> matplotlib.path.Path | np.ndarray:
    """
    Obtain the periodic interpolation of the existing path of a patch at a
    given resolution.

    Parameters
    -----------
    patch : matplotlib.patches.Patch
        Patch to obtain the original path from.
    resolution int
        Resolution at which to obtain the new path. The verticies of the new path
        will have shape (`resolution`, 2).

    Returns
    --------
    matplotlib.path.Path
        Interpolated path object.`
    """
    pth = patch.get_path()
    tfm = patch.get_transform()
    pathtfm = tfm.transform_path(pth)
    return interpolate_path(
        pathtfm, resolution=resolution, aspath=True, periodic=True, **kwargs
    )


def get_contour_paths(
    src: matplotlib.axes.Axes | matplotlib.contour.QuadContourSet,
    resolution: int = 100,
    minsize: int = 3,
    filter: bool = True,
) -> tuple[list[np.ndarray], list[str], list[dict]]:
    """
    Extract the paths of contours from a contour plot.

    Parameters
    ------------
    ax : matplotlib.axes.Axes | `matplotlib.contour.QuadContourSet`
        Axes to extract contours from.
    resolution : int
        Resolution of interpolated splines to return.
    filter : bool
        Whether to filter out paths which have no length.

    Returns
    --------
    contourspaths : list (list)
        List of lists, each represnting one line collection (a single contour). In the
        case where this contour is multimodal, there will be multiple paths for each
        contour.
    contournames : list
        List of names for contours, where they have been labelled, and there are no
        other text artists on the figure.
    contourstyles : list
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
    paths = [c for c in src._paths]

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

    if not filter:
        return interp_paths, names, edgecolors
    else:
        return (
            [p for p in interp_paths if p.size],
            [n for p, n in zip(interp_paths, names) if p.size],
            [c for p, c in zip(interp_paths, edgecolors) if p.size],
        )
