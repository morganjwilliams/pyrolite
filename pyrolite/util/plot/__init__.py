"""
Utility functions for working with matplotlib.

Parameters
----------
DEFAULT_CONT_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default continuous colormap.
DEFAULT_DICS_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default discrete colormap.
USE_PCOLOR : :class:`bool`
    Option to use the :func:`matplotlib.pyplot.pcolor` function in place
    of :func:`matplotlib.pyplot.pcolormesh`.
"""
from sys import platform

from ..log import Handle

logger = Handle(__name__)

from .density import USE_PCOLOR
from .style import DEFAULT_CONT_COLORMAP, DEFAULT_DISC_COLORMAP

FONTSIZE = 12

from .export import save_axes, save_figure

__all__ = ["save_figure", "save_axes", "DEFAULT_CONT_COLORMAP", "DEFAULT_DISC_COLORMAP"]
