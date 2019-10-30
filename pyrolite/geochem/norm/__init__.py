import os, sys
from pathlib import Path
import platform
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
import json
from ...comp import *
from ...util.pd import to_frame
from ...util.units import scale
from ...util.meta import pyrolite_datafolder
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__dbfile__ = pyrolite_datafolder(subfolder="geochem") / "refdb.json"


class Composition(object):
    def __init__(
        self, src, name=None, reference=None, reservoir=None, source=None, **kwargs
    ):
        self.comp = None
        self.units = None
        self.unc_2sigma = None

        self.name = name
        self.reference = reference
        self.reservoir = reservoir
        self.source = source

        self.filename = None
        self._df = None

        if isinstance(src, (str, Path)):
            self.filename = str(src)
            self._import_file(self.filename, **kwargs)
            self._process_imported_frame()
        elif isinstance(src, (pd.DataFrame, pd.Series)):  # composition dataframe
            self.comp = pd.DataFrame(
                src.loc[src.index[0], src.pyrochem.list_compositional].astype(np.float),
                index=["value"],
            )
        elif isinstance(src, dict):
            self._df = pd.DataFrame.from_dict(src).T
            self._process_imported_frame()
        else:
            raise NotImplementedError(
                "Import of compostions as {} not yet implemented.".format(type(src))
            )

        if (self.name is not None) and (self.filename is None):
            self.filename = "{}.csv".format(self.name)  # default naming

    def _import_file(self, filename, **kwargs):
        if filename.endswith(".csv"):
            self._df = pd.read_csv(filename, **kwargs).set_index("var").T
        elif filename.endswith("json"):
            self._df = pd.read_json(filename, **kwargs).set_index("var").T

    def _process_imported_frame(self):
        assert self._df is not None
        metadata = self._df.loc[
            "value", ["ModelName", "Reference", "Reservoir", "ModelType"]
        ].replace(np.nan, None)

        for src, dest in zip(
            ["ModelName", "Reference", "Reservoir", "ModelType"],
            ["name", "reference", "reservoir", "source"],
        ):
            setattr(self, dest, metadata[src])

        self.comp = self._df.loc[
            ["value"], self._df.pyrochem.list_compositional
        ].astype(np.float)
        self.comp = self.comp.dropna(axis=1)
        if "units" in self._df.index:
            self.units = self._df.loc["units", self.comp.columns]

        if "unc_2sigma" in self._df.index:
            self.unc_2sigma = self._df.loc["unc_2sigma", self.comp.columns].astype(
                np.float
            )

    def set_units(self, to="wt%"):
        """
        Set the units of the dataframe.

        Parameters
        ------------
        to : :class:`str`, :code:`"wt%"`
        """
        scales = self.units.apply(scale, target_unit=to).astype(np.float)
        self.comp *= scales
        self.units[:] = to
        return self

    def __getitem__(self, vars):
        """
        Allow access to model values via [] indexing e.g. Composition['Si', 'Cr'].

        Parameters
        -----------
        vars : :class:`str` | :class:`list`
            Variable(s) to get.
        """
        if isinstance(vars, (list, np.ndarray, pd.Index)):  # if iterable
            vars = [v if isinstance(v, str) else str(v) for v in vars]
        else:
            vars = [str(vars)]
        qry = self.comp.reindex(columns=vars).values.flatten()
        if len(qry) == 1:
            qry = qry[0]
        return qry

    def __str__(self):
        s = ""
        if self.name is not None:
            s += self.name + " "
        if self.reservoir is not None:
            s += "Model of " + self.reservoir + " "
        if self.reference is not None:
            s += "(" + self.reference + ")"
        return s

    def __repr__(self):
        r = self.__class__.__name__ + "("
        if self.filename is not None:
            r += "'{}'".format(Path(self.filename).name)
        for par in ["name", "reference", "reservoir", "source"]:
            if getattr(self, par) is not None:
                r += (
                    ",\n"
                    + " " * (len(self.__class__.__name__) + 1)
                    + "{}='{}'".format(par, getattr(self, par))
                )
        r += ")"
        return r


def all_reference_compositions(path=None):
    """
    Get a dictionary of all reference compositions indexed by name.

    Parameters
    -----------
    path : :class:`str` | :class:`pathlib.Path`

    Returns
    --------
    :class:`dict`
    """
    if path is None:
        path = __dbfile__
    with TinyDB(str(path)) as db:
        refs = {}
        for r in db.table().all():
            n, c = r["name"], r["composition"]
            refs[n] = Composition(json.loads(c), name=n)
        db.close()
    return refs


def get_reference_composition(name):
    """
    Retrieve a particular composition from the reference database.

    Parameters
    ------------
    name : :class:`str`
        Name of the reference composition model.

    Returns
    --------
    :class:`pyrolite.geochem.norm.Composition`
    """
    with TinyDB(str(__dbfile__)) as db:
        res = db.search(Query().name == name)
        db.close()
    assert len(res) == 1
    res = res[0]
    name, composition = res["name"], res["composition"]
    return Composition(json.loads(composition), name=name)


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


def update_database(path=None, encoding="cp1252", **kwargs):
    """
    Update the reference composition database.

    Notes
    ------
    This will take all csv files from the geochem/refcomp pyrolite data folder
    and construct a document-based JSON database.
    """
    if path is None:
        path = __dbfile__
    with TinyDB(str(path)) as db:
        db.purge()

        for f in get_reference_files():
            C = Composition(f, encoding=encoding, **kwargs)
            db.insert(
                {"name": C.name, "composition": C._df.T.to_json(force_ascii=False)}
            )
        db.close()
