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

if platform == "darwin":
    logger.debug('Using TkAgg renderer for Mac.')
    import matplotlib
    matplotlib.use('TkAgg')

from .style import DEFAULT_CONT_COLORMAP, DEFAULT_DISC_COLORMAP
from .density import USE_PCOLOR

FONTSIZE = 12

from .export import save_figure, save_axes

__all__ = ["save_figure", "save_axes", "DEFAULT_CONT_COLORMAP", "DEFAULT_DISC_COLORMAP"]
