import logging
from .mineral import Mineral

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


# Todo: class attributes to aggregate all collected minerals

__db__ = Mineral.db
