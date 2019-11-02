import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests

from pyrolite.util.pd import *
from pyrolite.util.georoc import *

import logging

root = logging.getLogger()
root.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(' %(name)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


bulk_GEOROC_download(reservoirs=['OIB'], redownload=False)
