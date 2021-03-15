"""
A utility submodule for standardised plot templates to be added to matplotlib axes.

Todo
----
* Make use of new ax.axline features (https://matplotlib.org/3.3.1/users/whats_new.html#new-axes-axline-method)
"""

from .pearce import pearceThNbYb, pearceTiNbYb
from .TAS import TAS
from ...util.log import Handle

logger = Handle(__name__)

__all__ = ["pearceThNbYb", "pearceTiNbYb", "TAS"]
