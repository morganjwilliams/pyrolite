"""
Baisc spatial utility functions.
"""

import numpy as np
import pandas as pd
import functools
import itertools
from psutil import virtual_memory  # memory check
from .math import on_finite
from .log import Handle

logger = Handle(__name__)


def _get_sqare_grid_segment_indicies(size, segments):
    """
    Get the indexes for segment boundaries for iterating over a grid within an array.

    Parameters
    ----------
    size : :class:`int`
        Shape of the square array.
    segments : :class:`int`
        Number of segments for the grid.

    Returns
    --------
    :class:`numpy.ndarray`
    """
    seg_size = size // segments
    segx = [(seg_size * ix, seg_size * (ix + 1)) for ix in range(segments)]
    segx[-1] = (seg_size * (segments - 1), size - 1)
    return [[*a, *b] for a, b in itertools.product(segx, segx)]


def _spherical_law_cosinse_GC_distance(φ1, φ2, λ1, λ2):
    """
    Spherical law of cosines calculation of distance between two points. Suffers from
    rounding errors for closer points.

    Parameters
    ----------
    φ1, φ2, λ1, λ2
        Numpy array wih latitudes and longitudes [x1, x2, y1, y2]
    """

    Δλ = np.abs(λ1 - λ2)
    # Δφ = np.abs(φ1 - φ2)
    return np.arccos(np.sin(φ1) * np.sin(φ2) + np.cos(φ1) * np.cos(φ2) * np.cos(Δλ))


def _vicenty_GC_distance(φ1, φ2, λ1, λ2):
    """
    Vicenty formula for an ellipsoid with equal major and minor axes.

    Vincenty T (1975) Direct and Inverse Solutions of Geodesics on the Ellipsoid with
    Application of Nested Equations. Survey Review 23:88–93.
    doi: 10.1179/SRE.1975.23.176.88

    Parameters
    ----------
    φ1, φ2 : :class:`numpy.ndarray`
        Numpy arrays wih latitudes.
    λ1, λ2 : :class:`numpy.ndarray`
        Numpy arrays wih longitude.
    """
    Δλ = np.abs(λ1 - λ2)
    # Δφ = np.abs(φ1 - φ2)

    _S = np.sqrt(
        (np.cos(φ2) * np.sin(Δλ)) ** 2
        + (np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)) ** 2
    )
    _C = np.sin(φ1) * np.sin(φ2) + np.cos(φ1) * np.cos(φ2) * np.cos(Δλ)
    return np.abs(np.arctan2(_S, _C))


def _haversine_GC_distance(φ1, φ2, λ1, λ2):
    """
    Haversine formula for great circle distance. Suffers from rounding errors for
    antipodal points.

    Parameters
    ----------
    φ1, φ2 : :class:`numpy.ndarray`
        Numpy arrays wih latitudes.
    λ1, λ2 : :class:`numpy.ndarray`
        Numpy arrays wih longitude.

    """
    Δλ = np.abs(λ1 - λ2)
    Δφ = np.abs(φ1 - φ2)
    return 2 * np.arcsin(
        np.sqrt(np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2)
    )


def _segmented_spatial_distance_matrix(
    φ1, φ2, λ1, λ2, metric, dtype="float32", segs=10
):
    size = np.max([a.shape[0] for a in [φ1, φ2, λ1, λ2]])
    angle = np.zeros((size, size), dtype=dtype)  # full matrix
    for ix_s, ix_e, iy_s, iy_e in _get_sqare_grid_segment_indicies(size, segs):
        angle[ix_s:ix_e, iy_s:iy_e] = metric(
            φ1[ix_s:ix_e][:, np.newaxis],
            φ2[iy_s:iy_e][np.newaxis, :],
            λ1[ix_s:ix_e][:, np.newaxis],
            λ2[iy_s:iy_e][np.newaxis, :],
        )
    return angle


def great_circle_distance(
    a,
    b=None,
    absolute=False,
    degrees=True,
    r=6371.0088,
    method=None,
    dtype="float32",
    max_memory_fraction=0.25,
):
    """
    Calculate the great circle distance between two lat, long points.

    Parameters
    ----------
    a, b : :class:`float` | :class:`numpy.ndarray`
        Lat-Long points or arrays to calculate distance between. If only one array is
        specified, a full distance matrix (i.e. calculate a point-to-point distance
        for every combination of points) will be returned.
    absolute : :class:`bool`, :code:`False`
        Whether to return estimates of on-sphere distances [True], or simply return the
        central angle between the points.
    degrees : :class:`bool`, :code:`True`
        Whether lat-long coordinates are in degrees [True] or radians [False].
    r : :class:`float`
        Earth radii for estimating absolute distances.
    method : :class:`str`, :code:`{'vicenty', 'cosines', 'haversine'}`
        Which method to use for great circle distance calculation. Defaults to the
        Vicenty formula.
    dtype : :class:`numpy.dtype`
        Data type for distance arrays, to constrain memory management.
    max_memory_fraction : :class:`float`
        Constraint to switch to calculating mean distances where :code:`matrix=True`
        and the distance matrix requires greater than a specified fraction of total
        avaialbe physical memory.
    """
    a = np.atleast_2d(np.array(a).astype(dtype))
    matrix = False
    if b is not None:
        b = np.atleast_2d(np.array(b).astype(dtype))
    else:
        matrix = True
        b = a.copy()

    # check the sizes of a and b - they should be the same

    if degrees:  # convert from degrees if needed
        a, b = np.deg2rad(a), np.deg2rad(b)

    φ1, φ2 = a[:, 0], b[:, 0]  # latitudes
    λ1, λ2 = a[:, 1], b[:, 1]  # longitudes

    if method is None:
        f = _vicenty_GC_distance
    else:
        if method.lower().startswith("cos"):
            f = _spherical_law_cosinse_GC_distance
        elif method.lower().startswith("hav"):
            f = _haversine_GC_distance
        else:  # Default to most precise
            f = _vicenty_GC_distance

    if matrix:
        # if matrix mode we need to turn these 1d arrays into 2d
        # but, with large arrays it'll spit out a memory error
        # so instead we can try to build it numerically
        size = np.max([a.shape[0] for a in [φ1, φ2, λ1, λ2]])
        mem = virtual_memory().total  # total physical memory available
        estimated_matrix_size = np.array([[1.0]], dtype=dtype).nbytes * size ** 2
        logger.debug(
            "Attempting to build {}x{} array of size {:.2f} Gb.".format(
                size, size, estimated_matrix_size / 1024 ** 3
            )
        )
        if estimated_matrix_size > (mem * max_memory_fraction):
            logger.warn(
                "Angle array for segmented distance matrix larger than maximum memory "
                "fraction, computing mean global distances instead."
            )
            angle = np.zeros((size, 1))
            # compute sum-distances for each lat-long pair
            for ix, (_φ1, _λ1) in enumerate(np.vstack([φ1, λ1])):
                angle[ix, 0] = f(_φ1, φ2, _λ1, λ2,)
        else:
            try:
                angle = np.atleast_1d(
                    f(
                        φ1[:, np.newaxis],
                        φ2[np.newaxis, :],
                        λ1[:, np.newaxis],
                        λ2[np.newaxis, :],
                    )
                )
            except (MemoryError, ValueError):
                logger.warn(
                    "Cannot directly compute distance matrix, attempting segmented distance"
                    " matrix instead."
                )
                # could set segs such that there is a maximum amount of memory per seg
                angle = _segmented_spatial_distance_matrix(φ1, φ2, λ1, λ2, f)
    else:
        angle = np.atleast_1d(f(φ1, φ2, λ1, λ2))

        if (
            np.isnan(angle).any() and f != _vicenty_GC_distance
        ):  # fallback for cos failure @ 0.
            fltr = np.isnan(angle)
            angle[fltr] = _vicenty_GC_distance(a[fltr, :], b[fltr, :])

    if absolute:
        return np.rad2deg(angle) * r
    else:
        return np.rad2deg(angle)


def piecewise(segment_ranges: list, segments=2, output_fmt=np.float):
    """
    Generator to provide values of quantizable paramaters which define a grid,
    here used to split up queries from databases to reduce load.

    Parameters
    ----------
    segment_ranges : :class:`list`
        List of segment ranges to create a grid from.
    segments : :class:`int`
        Number of segments.
    output_fmt
        Function to call on the output.
    """
    outf = np.vectorize(output_fmt)
    if isinstance(segments, np.int):
        segments = list(np.ones(len(segment_ranges), dtype=np.int) * segments)
    else:
        pass
    seg_width = [
        (x2 - x1) / segments[ix]  # can have negative steps
        for ix, (x1, x2) in enumerate(segment_ranges)
    ]
    separators = [
        np.linspace(x1, x2, segments[ix] + 1)[:-1]
        for ix, (x1, x2) in enumerate(segment_ranges)
    ]
    pieces = list(itertools.product(*separators))
    for piece in pieces:
        piece = np.array(piece)
        out = np.vstack((piece, piece + np.array(seg_width)))
        yield outf(out)


def spatiotemporal_split(
    segments=4,
    nan_lims=[np.nan, np.nan],
    # usebounds=False,
    # order=['minx', 'miny', 'maxx', 'maxy'],
    **kwargs
):
    """
    Creates spatiotemporal grid using piecewise function and arbitrary
    ranges for individial kw-parameters (e.g. age=(0., 450.)), and
    sequentially returns individial grid cell attributes.

    Parameters
    ----------
    segments : :class:`int`
        Number of segments.
    nan_lims : :class:`list` | :class:`tuple`
        Specificaiton of NaN indexes for missing boundaries.

    Yields
    -------
    :class:`dict`
        Iteration through parameter sets for each cell of the grid.
    """
    part = 0
    for item in piecewise(kwargs.values(), segments=segments):
        x1s, x2s = item
        part += 1
        params = {}
        for vix, var in enumerate(kwargs.keys()):
            vx1, vx2 = x1s[vix], x2s[vix]
            params[var] = (vx1, vx2)

        items = dict(
            south=params.get("lat", nan_lims)[0],
            north=params.get("lat", nan_lims)[1],
            west=params.get("long", nan_lims)[0],
            east=params.get("long", nan_lims)[1],
        )
        if "age" in params:
            items.update(
                dict(
                    minage=params.get("age", nan_lims)[0],
                    maxage=params.get("age", nan_lims)[1],
                )
            )

        items = {k: v for (k, v) in items.items() if not np.isnan(v)}
        # if usebounds:
        #    bounds = NSEW_2_bounds(items, order=order)
        #    yield bounds
        # else:
        yield items


def NSEW_2_bounds(cardinal, order=["minx", "miny", "maxx", "maxy"]):
    """
    Translates cardinal points to xy points in the form of bounds.
    Useful for converting to the format required for WFS from REST
    style queries.

    Parameters
    ----------
    cardinal : :class:`dict`
        Cardinally-indexed point bounds.
    order : :class:`list`
        List indicating order of returned x-y bound coordinates.

    Returns
    -------
    :class:`list`
        x-y indexed extent values in the specified order.

    """
    tnsltr = {
        xy: c
        for xy, c in zip(
            ["minx", "miny", "maxx", "maxy"], ["west", "south", "east", "north"]
        )
    }
    bnds = [cardinal.get(tnsltr[o]) for o in order]
    return bnds


def levenshtein_distance(seq_one, seq_two):
    """
    Compute the Levenshtein Distance between two sequences with comparable items.
    Adapted from Wiki pseudocode.

    Parameters
    ----------
    seq_one, seq_two : :class:`str` | :class:`list`
        Sequences to compare.

    Returns
    --------
    :class:`int`
    """
    m, n = len(seq_one), len(seq_two)
    D = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        D[i, 0] = i

    for j in range(n + 1):
        D[0, j] = j

    for j in np.arange(1, n + 1):  # n along columns
        for i in np.arange(1, m + 1):  # m along rows

            if seq_one[i - 1] == seq_two[j - 1]:
                substitutionCost = 0
            else:
                substitutionCost = 1

            D[i, j] = min(
                D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + substitutionCost
            )
    return D[-1, -1]
