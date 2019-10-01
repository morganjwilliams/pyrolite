"""
Submodule for working with compositional data.
"""

import pandas as pd
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from .codata import *
from .aggregate import *
from .impute import *

# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrocomp")
@pd.api.extensions.register_dataframe_accessor("pyrocomp")
class pyrocomp(object):
    """
    Custom dataframe accessor for pyrolite compositional transforms.
    """

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def renormalise(self, components: list = [], scale=100.0):
        """
        Renormalises compositional data to ensure closure.

        Parameters
        ------------
        components : :class:`list`
            Option subcompositon to renormalise to 100. Useful for the use case
            where compostional data and non-compositional data are stored in the
            same dataframe.
        scale : :class:`float`, :code:`100.`
            Closure parameter. Typically either 100 or 1.

        Returns
        --------
        :class:`pandas.DataFrame`
            Renormalized dataframe.
        """
        obj = self._obj
        return renormalise(obj, components=components, scale=scale)
