"""
A utility submodule for standardised plot templates to be added to matplotlib axes.

Todo
----
* Make use of new ax.axline features (https://matplotlib.org/3.3.1/users/whats_new.html#new-axes-axline-method)
"""

from ...util.log import Handle
from .feldspar import FeldsparTernary
from .jensen import JensenPlot
from .pearce import pearceThNbYb, pearceTiNbYb
from .QAP import QAP
from .TAS import TAS
from .USDA_soil_texture import USDASoilTexture

logger = Handle(__name__)

__all__ = [
    "pearceThNbYb",
    "pearceTiNbYb",
    "JensenPlot",
    "TAS",
    "USDASoilTexture",
    "QAP",
    "FeldsparTernary",
]
