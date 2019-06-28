import sys
import logging

# http://docs.python-guide.org/en/latest/writing/logging/
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.captureWarnings(True)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
