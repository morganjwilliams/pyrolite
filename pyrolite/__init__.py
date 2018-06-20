import sys
import logging
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
