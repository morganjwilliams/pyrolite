import pandas as pd
import logging
from ...util.meta import pyrolite_datafolder
from ...comp.codata import renormalise

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__folder__ = pyrolite_datafolder(subfolder="Aitchison")


def _load_frame(filename):
    df = pd.read_csv(__folder__ / filename)
    df.loc[:, ["A", "B", "C", "D", "E"]] = df.loc[
        :, ["A", "B", "C", "D", "E"]
    ].renormalise() # some of these are not closed to 100%
    df = df.set_index("Specimen")
    return df


def load_boxite():
    return _load_frame("boxite.csv")


def load_coxite():
    return _load_frame("coxite.csv")


def load_hongite():
    return _load_frame("hongite.csv")


def load_kongite():
    return _load_frame("kongite.csv")
