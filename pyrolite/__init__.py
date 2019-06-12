import sys
import logging

# http://docs.python-guide.org/en/latest/writing/logging/
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.captureWarnings(True)

from ._version import get_versions
from .plot import pyroplot  # import after logger setup to suppress numpydoc warnings

__version__ = get_versions()["version"]
del get_versions

__all__ = ["plot", "comp", "geochem", "mineral"]
