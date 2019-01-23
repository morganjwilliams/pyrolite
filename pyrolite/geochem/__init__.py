import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# direct imports for backwards compatibility
#from .geochem import *
#from .ind import *
#from .magma import *
#from .transform import *
#from .validate import *
