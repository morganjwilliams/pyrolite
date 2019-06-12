"""
This submodule contains interfaces and components for external software packages and
data repositories which pyrolite is not affliated with.
"""


import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

from . import datarepo
from . import alphamelts

__all__ = ['alphamelts', 'datarepo']
