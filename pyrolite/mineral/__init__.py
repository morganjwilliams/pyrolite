"""
A submodule for working with mineral data.
"""
import logging
from .mineral import Mineral
# from .db import * # import here creates the mineral database
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


# __db__ = Mineral.db
