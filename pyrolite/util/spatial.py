import numpy as np
import itertools
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()

def piecewise(segment_ranges:list, segments=2, output_fmt=np.float):
    """
    Generator to provide values of quantizable paramaters which define a grid,
    here used to split up queries from databases to reduce load.
    """
    outf = np.vectorize(output_fmt)
    if type(segments)==np.int:
        segments = list(np.ones(len(segment_ranges)) * segments)
    else:
        pass
    seg_width = [(x2 - x1) / segments[ix]  # can have negative steps
                 for ix, (x1, x2) in enumerate(segment_ranges)]
    separators = [np.linspace(x1, x2, segments[ix]+1)[:-1]
                  for ix, (x1, x2) in enumerate(segment_ranges)]
    pieces = list(itertools.product(*separators))
    for ix, i in enumerate(pieces):
        i = np.array(i)
        out = np.vstack((i, i + np.array(seg_width)))

        yield outf(out)


def spatiotemporal_split(segments=4,
                         nan_lims=[np.nan, np.nan],
                         #usebounds=False,
                         #order=['minx', 'miny', 'maxx', 'maxy'],
                         **kwargs):
    """
    Creates spatiotemporal grid using piecewise function and arbitrary
    ranges for individial kw-parameters (e.g. age=(0., 450.)), and
    sequentially returns individial grid cell attributes.
    """
    part = 0
    for item in piecewise(kwargs.values(), segments=segments):
        x1s, x2s = item
        part +=1
        params = {}
        for vix, var in enumerate(kwargs.keys()):
            vx1, vx2 = x1s[vix], x2s[vix]
            params[var] = (vx1, vx2)

        items = dict(south=params.get('lat', nan_lims)[0],
                     north=params.get('lat', nan_lims)[1],
                     west=params.get('long', nan_lims)[0],
                     east=params.get('long', nan_lims)[1],
                     )
        if 'age' in params:
            items.update(dict(minage=params.get('age', nan_lims)[0],
                              maxage=params.get('age', nan_lims)[1],))

        items = {k: v for (k, v) in items.items()
                 if not np.isnan(v)}
        #if usebounds:
        #    bounds = NSEW_2_bounds(items, order=order)
        #    yield bounds
        #else:
        yield items


def NSEW_2_bounds(cardinal, order=['minx', 'miny', 'maxx', 'maxy']):
    """
    Translates cardinal points to xy points in the form of bounds.
    Useful for converting to the format required for WFS from REST
    style queries.
    """
    tnsltr = {xy: c for xy, c in zip(['minx', 'miny', 'maxx', 'maxy'],
                                     ['west', 'south', 'east', 'north'])}
    bnds = [cardinal.get(tnsltr[o]) for o in order]
    return bnds
