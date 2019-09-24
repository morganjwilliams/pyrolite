import os, sys
from pathlib import Path
import platform
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
from ...comp import *
from ...util.pd import to_frame
from ...util.units import scale
from ...util.meta import pyrolite_datafolder
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class RefComp(object):
    """
    Reference compositional model object, principally used for normalisation.

    Todo
    ----
        * Ensure correct types are returned - i.e. floats vs. objects
    """

    def __init__(self, filename, **kwargs):
        self.filename = str(filename)
        self.data = pd.read_csv(filename, **kwargs)
        self.data = self.data.set_index("var")
        self.original_data = self.data.copy(deep=True)  # preserve unaltered record
        # self.add_oxides()
        self.collect_vars()
        self.set_units()

    def collect_vars(
        self,
        headers=["Reservoir", "Reference", "ModelName", "ModelType"],
        floatvars=["value", "unc_2sigma", "constraint_value"],
    ):
        """
        Build a data dictionary from the variable table.

        Parameters
        -----------
        headers : :class:`list`
            Headers to be treated separately to compositonal data.
        floatvars : :class:`list`
            Headers to be treated as compositonal data.
        """
        # integrate header data
        for h in headers:
            setattr(self, h, self.data.loc[h, "value"])

        self.vars = [
            i
            for i in self.data.index
            if (not pd.isna(self.data.loc[i, "value"])) and (i not in headers)
        ]
        self.data.loc[self.vars, floatvars] = self.data.loc[self.vars, floatvars].apply(
            pd.to_numeric, errors="coerce"
        )

    def set_units(self, to="ppm"):
        """
        Set the units of the dataframe.

        Parameters
        ------------
        to : :class:`str`, :code:`"ppm"`
        """
        v = self.vars
        self.data.loc[v, "scale"] = self.data.loc[v, "units"].apply(
            scale, target_unit=to
        )
        self.data.loc[v, "units"] = to
        self.data.loc[v, "value"] = self.data.loc[v, "value"] * self.data.loc[
            v, "scale"
        ].astype(np.float)

    def normalize(self, df):
        """
        Normalize the values within a dataframe to the refererence composition.
        Here we create indexes for normalisation of values.

        Parameters
        -----------
        df : :class:`pandas.DataFrame`
            Dataframe to normalize.

        Returns
        --------
        :class:`pandas.DataFrame`
            Normalised dataframe.

        Todo
        -----
            * Implement normalization of auxilary columns (LOD, uncertanties),
              potentially identified by lambda functions
              (e.g. :code:`lambda x: delim.join([str(x), "LOD"])`).
            * Uncertainty propogation
        """
        dfc = to_frame(df.copy(deep=True))

        cols = [c for c in dfc.columns if c in self.vars]
        _cols = set(cols)
        if len(cols) != len(_cols):
            msg = "Duplicated columns in dataframe."
            logger.warn(msg)
        cols = list(_cols)

        divisor = self.data.loc[cols, "value"].values

        dfc.loc[:, cols] = np.divide(dfc.loc[:, cols].values, divisor)
        return dfc

    def denormalize(self, df):
        """
        Un-normalize the values within a dataframe back to true composition.

        Parameters
        -----------
        df : :class:`pandas.DataFrame` | :class:`pandas.Series`
            Dataframe to de-normalize.

        Returns
        --------
        :class:`pandas.DataFrame`
            De-normalized dataframe.

        Todo
        -----
            * Implement normalization of auxilary columns (LOD, uncertanties),
              potentially identified by lambda functions
              (e.g. :code:`lambda x: delim.join([str(x), "LOD"])`).
            * Uncertainty propogation
        """
        dfc = to_frame(df.copy(deep=True))

        cols = [c for c in dfc.columns if c in self.vars]
        _cols = set(cols)
        if len(cols) != len(_cols):
            msg = "Duplicated columns in dataframe."
            logger.warn(msg)
        cols = list(_cols)

        multiplier = self.data.loc[cols, "value"].values

        dfc.loc[:, cols] *= multiplier
        return dfc

    def ratio(self, ratio):
        """
        Calculates an elemental ratio.

        Parameters
        ------------
        ratio : :class:`str`
            Slash-separated numerator and denominator for specific ratio.

        Returns
        --------
        :class:`float`
            Ratio, if it exists, otherwise :class:`np.nan`
        """
        try:
            assert "/" in ratio
            num, den = ratio.split("/")
            return self.data.loc[num, "value"] / self.data.loc[den, "value"]
        except:
            return np.nan

    def __getattr__(self, var):
        """
        Allow access to model values via attribute e.g. Model.Si

        Note
        ------
        This interferes with dataframe methods.

        Parameters
        -----------
        var : :class:`str`
            Variable to get.
        """
        if not isinstance(var, str):
            var = str(var)
        if var in self.data.index:
            return self.data.loc[var, "value"]
        else:
            return np.nan

    def __getitem__(self, vars):
        """
        Allow access to model values via [] indexing e.g. Model['Si', 'Cr'].
        Currently not implemented for ratios.

        Parameters
        -----------
        vars : :class:`str` | :class:`list`
            Variable(s) to get.
        """
        if (
            isinstance(vars, list)  # if iterable
            or isinstance(vars, pd.Index)
            or isinstance(vars, np.ndarray)
        ):
            vars = [v if isinstance(v, str) else str(v) for v in vars]
        elif not isinstance(vars, str):
            vars = str(vars)
        return self.data.loc[vars, ["value", "unc_2sigma", "units"]]

    def __str__(self):
        return "Model of " + self.Reservoir + " (" + self.Reference + ")"

    def __repr__(self):
        return (
            "RefComp("
            + Path(self.filename).name
            + ") from "
            + str(self.Reference)
            + "."
        )


def ReferenceCompositions(directory=None, formats=["csv"], **kwargs):
    """
    Build all reference models in a given directory. Here we use either the input
    directory, or the default data directory within this module.

    Parameters
    ----------
    directory : :class:`str`, :code:`None`
        Location of reference data files.
    formats : :class:`list`, :code:`["csv"]`
        List of potential data formats to draw from.
        Currently only csv will work.

    Returns
    --------
    :class:`dict`
        Dictionary of reference compositions.
    """
    # if platform.system() == "Windows":
    #    kwargs["encoding"] = kwargs.get("encoding", None) or "cp1252"
    # else:
    kwargs["encoding"] = kwargs.get("encoding", None) or "cp1252"

    curr_dir = os.path.realpath(__file__)
    module_dir = Path(sys.modules["pyrolite"].__file__).parent
    directory = directory or (pyrolite_datafolder(subfolder="geochem") / "refcomp")

    assert directory.exists() and directory.is_dir()

    files = []
    for fmt in formats:
        files.extend(directory.glob("./*." + fmt))

    comps = {}
    for f in files:
        r = RefComp(f, **kwargs)
        comps[r.ModelName] = r
    return comps


def get_reference_composition(name):
    return ReferenceCompositions()[name]


def get_reference_files(directory=None, formats=["csv"]):
    """
    Get a list of the reference composition files.

    Parameters
    -----------
    directory : :class:`str`, :code:`None`
        Location of reference data files.
    formats : :class:`list`, :code:`["csv"]`
        List of potential data formats to draw from. Currently only csv will work.

    Returns
    --------
    :class:`list`
    """
    directory = directory or (pyrolite_datafolder(subfolder="geochem") / "refcomp")
    assert directory.exists() and directory.is_dir()
    files = []
    for fmt in formats:
        files.extend(directory.glob("./*." + fmt))
    return files


def update_database(**kwargs):
    """
    Update the reference composition database.

    Note
    ------
    This will take all csv files from the geochem/refcomp pyrolite data folder
    and construct a document-based JSON database.
    """
    kwargs["encoding"] = kwargs.get("encoding", None) or "cp1252"

    dbfile = pyrolite_datafolder(subfolder="geochem") / "refdb.json"
    db = TinyDB(str(dbfile))
    db.purge()

    for f in get_reference_files():
        r = RefComp(f, **kwargs)
        r.ModelName = f.name
        db.insert({"name": r.ModelName, "composition": r.original_data.to_json()})
