import numpy as np
from numpy import cos, sin, deg2rad, arctan, arctan2, arcsin, arccos, sqrt, abs
import itertools
import logging
from pyrolite.util.math import isclose

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()


def _spherical_law_cosinse_GC_distance(ps):
    """
    Spherical law of cosines calculation of distance between two points. Suffers from
    rounding errors for closer points.

    Parameters
    ----------
    ps
        Numpy array wih latitudes and longitudes [x1, x2, y1, y2]
    """
    φ1, φ2 = ps[2:]  # latitude
    λ1, λ2 = ps[:2]  # longitude
    Δλ = abs(λ1 - λ2)
    Δφ = abs(φ1 - φ2)
    return arccos(sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ))


def _vicenty_GC_distance(ps):
    """
    Vicenty formula for an ellipsoid with equal major and minor axes.

    Vincenty T (1975) Direct and Inverse Solutions of Geodesics on the Ellipsoid with
    Application of Nested Equations. Survey Review 23:88–93.
    doi: 10.1179/SRE.1975.23.176.88

    Parameters
    ----------
    ps
        Numpy array wih latitudes and longitudes [x1, x2, y1, y2]
    """
    φ1, φ2 = ps[2:]  # latitude
    λ1, λ2 = ps[:2]  # longitude
    Δλ = abs(λ1 - λ2)
    Δφ = abs(φ1 - φ2)

    _S = sqrt(
        (cos(φ2) * sin(Δλ)) ** 2
        + (cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(Δλ)) ** 2
    )
    _C = sin(φ1) * sin(φ2) + cos(φ1) * cos(φ2) * cos(Δλ)
    return np.abs(arctan2(_S, _C))


def _haversine_GC_distance(ps):
    """
    Haversine formula for great circle distance. Suffers from rounding errors for
    antipodal points.

    Parameters
    ----------
    ps
        Numpy array wih latitudes and longitudes [x1, x2, y1, y2]
    """
    φ1, φ2 = ps[2:]  # latitude
    λ1, λ2 = ps[:2]  # longitude
    Δλ = abs(λ1 - λ2)
    Δφ = abs(φ1 - φ2)
    return 2 * arcsin(sqrt(sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2))


def great_circle_distance(
    p1, p2, absolute=False, degrees=True, r=6371.0088, method=None
):
    """
    Calculate the great circle distance between two lat, long points.

    Parameters
    ----------
    p1, p2
        Lat-Long points to calculate distance between.
    absolute : bool, False
        Whether to return estimates of on-sphere distances [True], or simply return the
        central angle between the points.
    degrees : bool, True
        Whether lat-long coordinates are in degrees [True] or radians [False].
    r : np.number
        Earth radii for estimating absolute distances.
    method : {'vicenty', 'cosines', 'haversine'}
        Which method to use for great circle distance calculation. Defaults to the
        Vicenty formula.
    """
    x1, y1 = p1
    x2, y2 = p2
    ps = np.array([x1, x2, y1, y2]).astype(np.float)

    if degrees:
        ps = deg2rad(ps)

    if method is None:
        f = _vicenty_GC_distance
    else:
        if method.lower().startswith("cos"):
            f = _spherical_law_cosinse_GC_distance
        elif method.lower().startswith("hav"):
            f = _haversine_GC_distance
        else:  # Default to most precise
            f = _vicenty_GC_distance

    angle = f(ps)

    if np.isnan(angle) and f != _vicenty_GC_distance:  # fallback for cos failure @ 0.
        angle = _vicenty_GC_distance(ps)

    if absolute:
        return np.rad2deg(angle) * r
    else:
        return np.rad2deg(angle)


def piecewise(segment_ranges: list, segments=2, output_fmt=np.float):
    """
    Generator to provide values of quantizable paramaters which define a grid,
    here used to split up queries from databases to reduce load.
    """
    outf = np.vectorize(output_fmt)
    if type(segments) == np.int:
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
    for ix, i in enumerate(pieces):
        i = np.array(i)
        out = np.vstack((i, i + np.array(seg_width)))

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
    """
    tnsltr = {
        xy: c
        for xy, c in zip(
            ["minx", "miny", "maxx", "maxy"], ["west", "south", "east", "north"]
        )
    }
    bnds = [cardinal.get(tnsltr[o]) for o in order]
    return bnds
