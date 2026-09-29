"""
Baisc spatial utility functions.
"""

from typing import Generator, Any, Any

import itertools
from collections.abc import Callable

import numpy as np

try:
    from psutil import virtual_memory  # memory check
except ImportError:
    virtual_memory = None

from .log import Handle

logger = Handle(__name__)


def _get_sqare_grid_segment_indicies(size: int, segments: int):
    """
    Get the indexes for segment boundaries for iterating over a grid within an array.

    Parameters
    ----------
    size : int
        Shape of the square array.
    segments : int
        Number of segments for the grid.

    Returns
    --------
    numpy.ndarray
    """
    seg_size = size // segments
    segx = [(seg_size * ix, seg_size * (ix + 1)) for ix in range(segments)]
    segx[-1] = (seg_size * (segments - 1), size - 1)
    return [[*a, *b] for a, b in itertools.product(segx, segx)]


def _spherical_law_cosinse_GC_distance(
    φ1: float | np.ndarray,
    φ2: float | np.ndarray,
    λ1: float | np.ndarray,
    λ2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Spherical law of cosines calculation of distance between two points. Suffers from
    rounding errors for closer points.

    Parameters
    ----------
    φ1, φ2, λ1, λ2 : float | numpy.ndarray
        Latitudes and longitudes [x1, x2, y1, y2]
    """

    Δλ = np.abs(λ1 - λ2)
    # Δφ = np.abs(φ1 - φ2)
    return np.arccos(np.sin(φ1) * np.sin(φ2) + np.cos(φ1) * np.cos(φ2) * np.cos(Δλ))


def _vicenty_GC_distance(
    φ1: float | np.ndarray,
    φ2: float | np.ndarray,
    λ1: float | np.ndarray,
    λ2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Vicenty formula for an ellipsoid with equal major and minor axes.

    Vincenty T (1975) Direct and Inverse Solutions of Geodesics on the Ellipsoid with
    Application of Nested Equations. Survey Review 23:88-93.
    doi: 10.1179/SRE.1975.23.176.88

    Parameters
    ----------
    φ1, φ2 : float | numpy.ndarray
        Numpy arrays wih latitudes.
    λ1, λ2 : float
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


def _haversine_GC_distance(
    φ1: float | np.ndarray,
    φ2: float | np.ndarray,
    λ1: float | np.ndarray,
    λ2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Haversine formula for great circle distance. Suffers from rounding errors for
    antipodal points.

    Parameters
    ----------
    φ1, φ2 : float | numpy.ndarray
        Numpy arrays wih latitudes.
    λ1, λ2 : float | numpy.ndarray
        Numpy arrays wih longitude.

    """
    Δλ = np.abs(λ1 - λ2)
    Δφ = np.abs(φ1 - φ2)
    return 2 * np.arcsin(
        np.sqrt(np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2)
    )


def _segmented_spatial_distance_matrix(
    φ1: np.ndarray,
    φ2: np.ndarray,
    λ1: np.ndarray,
    λ2: np.ndarray,
    metric: Callable,
    dtype: str | np.dtype = "float32",
    segs: int = 10,
):
    size: int = np.max([a.shape[0] for a in [φ1, φ2, λ1, λ2]])
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
    a: np.ndarray[tuple[int]] | np.ndarray[tuple[int, int]],
    b: np.ndarray[tuple[int]] | np.ndarray[tuple[int, int]] | None = None,
    absolute: bool = False,
    degrees: bool = True,
    r: float = 6371.0088,
    method: str | None = None,
    dtype: str | np.dtype = "float32",
    max_memory_fraction: float = 0.25,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """
    Calculate the great circle distance between two lat, long points.

    Parameters
    ----------
    a, b : numpy.ndarray
        Lat-Long points or arrays to calculate distance between. If only one array is
        specified, a full distance matrix (i.e. calculate a point-to-point distance
        for every combination of points) will be returned.
    absolute : bool
        Whether to return estimates of on-sphere distances [True], or simply return the
        central angle between the points.
    degrees : bool
        Whether lat-long coordinates are in degrees [True] or radians [False].
    r : float
        Earth radii for estimating absolute distances.
    method : str
        Which method to use for great circle distance calculation. Defaults to the
        Vicenty formula.
    dtype : numpy.dtype
        Data type for distance arrays, to constrain memory management.
    max_memory_fraction : float
        Constraint to switch to calculating mean distances where `matrix=True`
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
        estimated_matrix_size = np.array([[1.0]], dtype=dtype).nbytes * size**2
        logger.debug(
            f"Attempting to build {size}x{size} array of size {estimated_matrix_size / 1024**3:.2f} Gb."
        )

        infeasible = (
            estimated_matrix_size > (virtual_memory().total * max_memory_fraction)
            if virtual_memory is not None
            else False
        )

        if infeasible:
            logger.warning(
                "Angle array for segmented distance matrix larger than maximum memory "
                "fraction, computing mean global distances instead."
            )
            angle = np.zeros((size, 1))
            # compute sum-distances for each lat-long pair
            for ix, (_φ1, _λ1) in enumerate(np.vstack([φ1, λ1])):
                angle[ix, 0] = f(_φ1, φ2, _λ1, λ2)
        else:
            try:
                angle = np.atleast_1d(
                    f(φ1[:, None], φ2[None, :], λ1[:, None], λ2[None, :])
                )
            except (MemoryError, ValueError):
                logger.warning(
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
            angle[fltr] = _vicenty_GC_distance(φ1[fltr], φ2[fltr], λ1[fltr], λ2[fltr])

    if absolute:
        return np.rad2deg(angle) * r
    else:
        return np.rad2deg(angle)


def piecewise(
    segment_ranges: list[tuple[float, float]],
    segments: int = 2,
    output_fmt: np.dtype | Callable = np.float64,
) -> Generator[np.ndarray, None, None]:
    """
    Generator to provide values of quantizable paramaters which define a grid,
    here used to split up queries from databases to reduce load.

    Parameters
    ----------
    segment_ranges : list
        List of segment ranges to create a grid from.
    segments : int
        Number of segments.
    output_fmt
        Function to call on the output.
    """
    outf = np.vectorize(output_fmt)
    if isinstance(segments, int):
        segments: list[int] = list(np.ones(len(segment_ranges), dtype=int) * segments)
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
    segments: int = 4,
    nan_lims: tuple[float, float] | None = None,
    # usebounds=False,
    # order=['minx', 'miny', 'maxx', 'maxy'],
    **kwargs,
) -> Generator[dict, None, None]:
    """
    Creates spatiotemporal grid using piecewise function and arbitrary
    ranges for individial kw-parameters (e.g. age=(0., 450.)), and
    sequentially returns individial grid cell attributes.

    Parameters
    ----------
    segments : int
        Number of segments.
    nan_lims :  tuple[float,float]
        Specificaiton of NaN indexes for missing boundaries.

    Yields
    -------
    dict
        Iteration through parameter sets for each cell of the grid.
    """
    if nan_lims is None:
        nan_lims = (np.nan, np.nan)
    part = 0
    for item in piecewise(list(kwargs.values()), segments=segments):
        x1s, x2s = item
        part += 1
        params = {}
        for vix, var in enumerate(kwargs.keys()):
            vx1, vx2 = x1s[vix], x2s[vix]
            params[var] = (vx1, vx2)

        items = {
            "south": params.get("lat", nan_lims)[0],
            "north": params.get("lat", nan_lims)[1],
            "west": params.get("long", nan_lims)[0],
            "east": params.get("long", nan_lims)[1],
        }
        if "age" in params:
            items.update(
                {
                    "minage": params.get("age", nan_lims)[0],
                    "maxage": params.get("age", nan_lims)[1],
                }
            )

        items = {k: v for (k, v) in items.items() if not np.isnan(v)}
        # if usebounds:
        #    bounds = NSEW_2_bounds(items, order=order)
        #    yield bounds
        # else:
        yield items


def NSEW_2_bounds(cardinal: dict, order: list | None = None) -> list:
    """
    Translates cardinal points to xy points in the form of bounds.
    Useful for converting to the format required for WFS from REST
    style queries.

    Parameters
    ----------
    cardinal : dict
        Cardinally-indexed point bounds.
    order : list
        List indicating order of returned x-y bound coordinates.

    Returns
    -------
    list
        x-y indexed extent values in the specified order.

    """
    if order is None:
        order = ["minx", "miny", "maxx", "maxy"]
    tnsltr = {
        xy: c
        for xy, c in zip(
            ["minx", "miny", "maxx", "maxy"], ["west", "south", "east", "north"]
        )
    }
    bnds = [cardinal.get(tnsltr[o]) for o in order]
    return bnds


def levenshtein_distance(seq_one: str | list[str], seq_two: str | list[str]) -> int:
    """
    Compute the Levenshtein Distance between two sequences with comparable items.
    Adapted from Wiki pseudocode.

    Parameters
    ----------
    seq_one, seq_two : str | list[str]
        Sequences to compare.

    Returns
    --------
    int
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
