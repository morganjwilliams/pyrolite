"""
Utility functions for working with matplotlib.

Todo:
------

    * Functions for working with and modifying legend entries.

        ax.lines + ax.patches + ax.collections + ax.containers, handle ax.parasites

Attributes
----------
DEFAULT_CONT_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default continuous colormap.
DEFAULT_DICS_COLORMAP : :class:`matplotlib.colors.ScalarMappable`
    Default discrete colormap.
USE_PCOLOR : :class:`bool`
    Option to use the :func:`matplotlib.pyplot.pcolor` function in place
    of :func:`matplotlib.pyplot.pcolormesh`.

"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from .style import DEFAULT_CONT_COLORMAP, DEFAULT_DISC_COLORMAP
from .density import USE_PCOLOR

FONTSIZE = 12

from .export import save_figure, save_axes

__all__ = ["save_figure", "save_axes", "DEFAULT_CONT_COLORMAP", "DEFAULT_DISC_COLORMAP"]
