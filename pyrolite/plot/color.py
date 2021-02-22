import copy
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from pyrolite.util.plot import DEFAULT_CONT_COLORMAP, DEFAULT_DISC_COLORMAP
from ..util.log import Handle

logger = Handle(__name__)

_face_edge_equivalents = {
    "facecolors": "edgecolors",
    "markerfacecolor": "markeredgecolor",
    "mfc": "mec",
}


def get_cmode(c=None):
    """
    Find which mode a color is supplied as, such that it can be processed.

    Parameters
    -----------
    c :  :class:`str` | :class:`list` | :class:`tuple` | :class:`numpy.ndarray`
        Color arguments as typically passed to :func:`matplotlib.pyplot.scatter`
        or :func:`matplotlib.pyplot.plot`.
    """
    cmode = None
    if c is not None:  # named | hex | rgb | rgba
        logger.debug("Checking singular color modes.")
        if isinstance(c, str):
            if c.startswith("#"):
                cmode = "hex"
            else:
                cmode = "named"
        elif isinstance(c, tuple):
            if len(c) == 3:
                cmode = "rgb"
            elif len(c) == 4:
                cmode = "rgba"
        else:
            pass

        if cmode is None:  # list | ndarray | ndarray(rgb) | ndarray(rgba)
            logger.debug("Checking array-based color modes.")
            if isinstance(c, (np.ndarray, list, pd.Series, pd.Index)):
                c = np.array(c, dtype=getattr(c, "dtype", "object"))
                convertible = False
                try:  # could test all of them, or just a few
                    _ = [matplotlib.colors.to_rgba(_c) for _c in [c[0], c[-1]]]
                    convertible = True
                except (ValueError, TypeError):  # string cannot be converted to color
                    pass
                if all([isinstance(_c, (np.ndarray, list, tuple)) for _c in c]):
                    # could have an error if you put in mixed rgb/rgba
                    if len(c[0]) == 3:
                        cmode = "rgb_array"
                    elif len(c[0]) == 4:
                        cmode = "rgba_array"
                    else:
                        pass
                elif all([isinstance(_c, str) for _c in c]):
                    if convertible:
                        if all([_c.startswith("#") for _c in c]):
                            cmode = "hex_array"
                        elif not any([_c.startswith("#") for _c in c]):
                            cmode = "named_array"
                        else:
                            cmode = "mixed_str_array"
                    else:
                        cmode = "categories"
                elif all([isinstance(_c, np.number) for _c in np.array(c).flatten()]):
                    cmode = "value_array"
                else:
                    if convertible:
                        cmode = "mixed_fmt_color_array"
                if cmode is None:
                    # default cmode to fall back on - e.g. list of tuples/intervals etc
                    cmode = "categories"
    if cmode is None:
        logger.debug("Color mode not found for item of type {}".format(type(c)))
        raise NotImplementedError  # single value, mixed numbers, strings etc
    else:
        logger.debug("Color mode recognized: {}".format(cmode))
        return cmode


def process_color(
    c=None,
    color=None,
    cmap=None,
    alpha=None,
    norm=None,
    cmap_under=(1, 1, 1, 0.0),
    color_converter=matplotlib.colors.to_rgba,
    color_mappings={},
    size=None,
    **otherkwargs,
):
    """
    Color argument processing for pyrolite plots, returning a standardised output.

    Parameters
    -----------
    c : :class:`str` | :class:`list` | :class:`tuple` | :class:`numpy.ndarray`
        Color arguments as typically passed to :func:`matplotlib.pyplot.scatter`.
    color : :class:`str` | :class:`list` | :class:`tuple` | :class:`numpy.ndarray`
        Color arguments as typically passed to :func:`matplotlib.pyplot.plot`
    cmap : :class:`str` | :class:`~matplotlib.cm.ScalarMappable`
        Colormap for mapping unknown color values.
    alpha : :class:`float`
        Alpha to modulate color opacity.
    norm : :class:`~matplotlib.colors.Normalize`
        Normalization for the colormap.
    cmap_under : :class:`str` | :class:`tuple`
        Color for values below the lower threshold for the cmap.
    color_converter
        Function to use to convert colors (from strings, hex, tuples etc).
    color_mappings : :class:`dict`
        Dictionary containing category-color mappings for individual color variables,
        with the default color mapping having the key 'color'. For use where
        categorical values are specified for a color variable.
    size : :class:`int`
        Size of the data array along the first axis.

    Returns
    --------
    C : :class:`tuple` | :class:`numpy.ndarray`
        Color returned in standardised RGBA format.

    Notes
    ------
    As formulated here, the addition of unused styling parameters may cause some
    properties (associated with 'c') to be set to None - and hence revert to defaults.
    This might be mitigated if the context could be checked - e.g. via checking
    keyword argument membership of :func:`~pyrolite.util.plot.style.scatterkwargs` etc.
    """
    assert not ((c is not None) and (color is not None))
    for kw in [  # extra color kwargs
        "facecolors",
        "markerfacecolor",
        "mfc",
        "markeredgecolor",
        "mec",
        "edgecolors",
        "ec",
        "linecolor",
        "lc",
        "ecolor",  # for errobar
        "facecolor",
    ]:
        if kw in otherkwargs:  # this allows processing of alpha with a given color
            _pc = process_color(
                c=otherkwargs[kw],
                alpha=alpha,
                cmap=cmap,
                norm=norm,
                color_mappings={"color": color_mappings.get(kw)},
            )
            otherkwargs[kw] = _pc.get("color", None)
    if c is not None:
        C = c
    elif color is not None:
        C = color
    else:  # neither color is specified
        d = {
            **{
                k: v
                for k, v in {
                    "c": c,
                    "color": color,
                    "cmap": cmap,
                    "norm": norm,
                    "alpha": alpha,
                }.items()
                if v is not None
            },
            **otherkwargs,
        }
        # the parameter 'c' will override 'facecolor' and related
        if any([k in d for k in _face_edge_equivalents.keys()]):
            d.pop("c", None)
        return d

    cmode = get_cmode(C)

    _c, _color = None, None
    if cmode in ["hex", "named", "rgb", "rgba"]:  # single color
        C = matplotlib.colors.to_rgba(C)
        if alpha is not None:
            C = (
                *C[:-1],
                alpha * C[-1],
            )  # can't assign to tuple, create new one instead
        _c, _color = np.array([C]), C  # Convert to standardised form
        if size is not None:
            _c = np.ones((size, 1)) * _c  # turn this into a full array as a fallback
    else:
        C = np.array(C)
        if cmode in [
            "hex_array",
            "named_array",
            "mixed_str_array",
        ]:
            C = np.array([matplotlib.colors.to_rgba(ic) for ic in C])
        elif cmode in ["rgb_array", "rgba_array"]:
            C = np.array([matplotlib.colors.to_rgba(ic) for ic in C])
        elif cmode in ["mixed_fmt_color_array"]:
            C = np.array([matplotlib.colors.to_rgba(ic) for ic in C])
        elif cmode in ["value_array"]:
            cmap = cmap or DEFAULT_CONT_COLORMAP
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            if cmap_under is not None:
                cmap = copy.copy(cmap)  # without this, it would modify the global cmap
                cmap.set_under(color=cmap_under)
            norm = norm or plt.Normalize(
                vmin=np.nanmin(np.array(C)), vmax=np.nanmax(np.array(C))
            )
            C = cmap(norm(C))
        elif cmode == "categories":
            uniqueC = np.unique(C)
            cmapper = color_mappings.get("color")
            if cmapper is None:
                _C = np.ones_like(C, dtype="int") * np.nan

                cmap = cmap or DEFAULT_DISC_COLORMAP
                if isinstance(cmap, str):
                    cmap = plt.get_cmap(cmap)

                for ix, cat in enumerate(uniqueC):
                    _C[C == cat] = ix / len(uniqueC)
                C = cmap(_C)
            else:
                unique_vals = np.array(list(cmapper.values()))
                _C = np.ones((len(C), 4), dtype=np.float)
                for cat in uniqueC:
                    val = matplotlib.colors.to_rgba(cmapper.get(cat))
                    _C[C == cat] = val  # get the mapping frome the dict
                C = _C
        else:
            pass
        if alpha is not None:
            C[:, -1] = alpha
        _c, _color = C, C

    d = {"color": _color, **otherkwargs}
    # the parameter 'c' will override 'facecolors' and related for markers
    if not any(
        [
            k in d
            for k in [item for args in _face_edge_equivalents.items() for item in args]
        ]
    ):
        d["c"] = _c
    else:
        # for each of the facecolor modes specified return an edge variant
        for face, edge in _face_edge_equivalents.items():
            if (face in d) and not (edge in d):
                d[edge] = _c
            if (edge in d) and not (face in d):
                d[face] = _c
    return d
