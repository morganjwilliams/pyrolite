import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from pyrolite.util.plot import DEFAULT_CONT_COLORMAP


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
            if isinstance(c, (np.ndarray, list)):
                c = np.array(c)
                if all([isinstance(_c, (np.ndarray, list, tuple)) for _c in c]):
                    # could have an error if you put in mixed rgb/rgba
                    if len(c[0]) == 3:
                        cmode = "rgb_array"
                    elif len(c[0]) == 4:
                        cmode = "rgba_array"
                    else:
                        pass
                elif all([isinstance(_c, str) for _c in c]):
                    if all([_c.startswith("#") for _c in c]):
                        cmode = "hex_array"
                    elif not any([_c.startswith("#") for _c in c]):
                        cmode = "named_array"
                    else:
                        cmode = "mixed_str_array"
                elif all([isinstance(_c, (np.float, np.int)) for _c in c]):
                    cmode = "value_array"
                else:
                    pass
    if cmode is None:
        raise NotImplementedError  # single value, mixed numbers, strings etc
    return cmode


def process_color(
    c=None,
    color=None,
    cmap=None,
    alpha=None,
    norm=None,
    cmap_under=(1, 1, 1, 0.0),
    color_converter=matplotlib.colors.to_rgba,
    **otherkwargs
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

    Returns
    --------
    C : :class:`tuple` | :class:`numpy.ndarray`
        Color returned in standardised RGBA format.

    Notes
    ---------

    This will not modify other named colors (e.g. linecolor, markercolor).
    """
    assert not ((c is not None) and (color is not None))

    if c is not None:
        C = c
    elif color is not None:
        C = color
    else:
        return {"c": c, "color": color, "cmap": cmap, "norm": norm, **otherkwargs}

    cmode = get_cmode(C)
    _c, _color = None, None
    if cmode in ["hex", "named", "rgb", "rgba"]:  # single color
        C = matplotlib.colors.to_rgba(C)
        if alpha is not None:
            C = (*C[:-1], alpha)  # can't assign to tuple, create new one instead
        _c, _color = np.array([C]), C  # Convert to standardised form
    else:
        if cmode in ["hex_array", "named_array", "mixed_str_array"]:
            C = np.array([matplotlib.colors.to_rgba(ic) for ic in C])
        elif cmode in ["rgb_array", "rgba_array"]:
            C = np.array([matplotlib.colors.to_rgba(ic) for ic in C])
        elif cmode in ["value_array"]:
            cmap = cmap or DEFAULT_CONT_COLORMAP
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            if cmap_under is not None:
                cmap.set_under(color=cmap_under)
            norm = norm or plt.Normalize(
                vmin=np.nanmin(np.array(c)), vmax=np.nanmax(np.array(c))
            )
            C = cmap(norm(C))
        else:
            pass
        if alpha is not None:
            C[:, -1] = alpha
        _c, _color = C, C
    return {"c": _c, "color": _color, **otherkwargs}


"""
from pyrolite.util.plot import scatterkwargs, linekwargs

plt.plot(
    np.random.randn(2),
    np.random.randn(2),
    lc=linekwargs(process_color(c=np.array([0, 1]), alpha=0.5))["color"],
)
"""
