"""
Submodule for working with geochemical data.
"""
import logging
import pandas as pd

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# direct imports for backwards compatibility
from .ind import *
from .magma import *
from .parse import *
from .transform import *
from .validate import *
from .alteration import *
from .norm import *

# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrochem")
@pd.api.extensions.register_dataframe_accessor("pyrochem")
class pyrochem(object):
    """
    Custom dataframe accessor for pyrolite geochemistry.
    """

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    def norm_to(self, norm_to=None):
        pass
