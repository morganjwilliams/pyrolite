"""
Reference compostitions and compositional normalisation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from ..util.log import Handle
from ..util.meta import pyrolite_datafolder
from ..util.text import to_width
from ..util.units import scale

logger = Handle(__name__)

__dbfile__ = pyrolite_datafolder(subfolder="geochem") / "refdb.json"


NORMDB: pd.DataFrame = pd.DataFrame()


def set_DB(path: str | Path):
    """
    Assign the database used for reference compositions, as per a given path.
    """
    with open(__dbfile__, "r") as f:
        global NORMDB
        NORMDB = pd.DataFrame(json.loads(f.read()))
        NORMDB["composition"] = NORMDB["composition"].map(json.loads)


set_DB(__dbfile__)


class Composition:
    def __init__(
        self,
        src: str | Path | pd.DataFrame | pd.Series | dict,
        name: str | None = None,
        reference: str | None = None,
        reservoir: str | None = None,
        source: str | None = None,
        **kwargs,
    ):
        """A composition with units and uncertainties for each compositional
        variable.

        Attributes
        -----------
        name : str
            Name of the composition.
        reference : str
            Reference for the composition.
        reservoir : str
            Optionally-specified reservoir for the specific compositoin (e.g. Primitive
            Mantle).
        source : `str
            Source of the composition (typically method of derivation,
            e.g. 'calculated').
        filename : str | pathlib.Path
            File which the composition is derived from.
        comp : pandas.DataFrame
            A 1-row dataframe
        units : pandas.Series
            Units of the compositional variables.
        unc_2sigma : pandas.Series
            Uncertainties for the compositional variables.
        """
        self.comp = None
        self.units = None
        self.unc_2sigma = None

        self.name: str | None = name
        self.reference: str | None = reference
        self.reservoir: str | None = reservoir
        self.source: str | None = source

        self.filename = None
        self._df = None

        if isinstance(src, (str, Path)):
            self.filename = str(src)
            self._import_file(self.filename)
            self._process_imported_frame()
        elif isinstance(src, (pd.DataFrame, pd.Series)):  # composition dataframe
            self.comp = pd.Series(
                src.loc[src.index[0], src.pyrochem.list_compositional].astype(float),
            )
        elif isinstance(src, dict):
            self._df: DataFrame = pd.DataFrame.from_dict(src)
            self._process_imported_frame()
        else:
            raise NotImplementedError(
                f"Import of compostions as {type(src)} not yet implemented."
            )

        if (self.name is not None) and (self.filename is None):
            self.filename = f"{self.name}.csv"  # default naming

    def _import_file(self, filename: str | Path, **kwargs):
        filename = Path(filename)
        if filename.suffix == ".csv":
            self._df = pd.read_csv(filename, **kwargs).set_index("var")
        elif filename.suffix == ".json":
            self._df = pd.read_json(filename, **kwargs).set_index("var")

    def _process_imported_frame(self):
        assert self._df is not None
        metadata: pd.Series = self._df.reindex(
            index=[
                "ModelName",
                "Reservoir",
                "ModelType",
                "Reference",
                "Citation",
                "DOI",
                "Description",
            ],
            columns=["value"],
        ).iloc[:, 0]
        metadata[pd.isnull(metadata)] = None
        for src, dest in zip(
            [
                "ModelName",
                "Reservoir",
                "ModelType",
                "Reference",
                "Citation",
                "DOI",
                "Description",
            ],
            [
                "name",
                "reservoir",
                "source",
                "reference",
                "citation",
                "doi",
                "description",
            ],
        ):
            setattr(self, dest, metadata.get(src, None))

        self.comp: pd.Series = (
            self._df["value"].pyrochem.compositional.astype(float).dropna()
        )
        if "units" in self._df.columns:
            self.units: pd.Series = self._df.loc[self.comp.index, "units"]

        if "unc_2sigma" in self._df.columns:
            self.unc_2sigma: pd.Series = self._df.loc[
                self.comp.index, "unc_2sigma"
            ].astype(float)

    def set_units(self, to: str = "wt%"):
        """
        Set the units of the dataframe.

        Parameters
        ------------
        to : str, `"wt%"`
        """
        scales: pd.Series = self.units.apply(scale, target_unit=to).astype(float)
        self.comp *= scales
        self.units = pd.Series([to] * len(self.units))
        return self

    def describe(self, verbose: bool = True, **kwargs) -> str:
        metadata = self._df.reindex(
            index=[
                "ModelName",
                "Reservoir",
                "ModelType",
                "Reference",
                "Citation",
                "DOI",
                "Description",
            ]
        )["value"].dropna()
        desc = ""
        if verbose:
            desc += str(self)
            desc += "\n"

        if "Description" in metadata:
            desc += metadata["Description"]
            desc += "\n"
        if "Citation" in metadata:
            desc += metadata["Citation"]
            if "DOI" in metadata:
                desc += " "
                desc += "doi: {}".format(metadata["DOI"])
        return to_width(desc, **kwargs)

    def __getitem__(self, variables: list | np.ndarray | pd.Index):
        """
        Allow access to model values via [] indexing e.g. Composition['Si', 'Cr'].

        Parameters
        ----------
        variables : str | list
            Variable(s) to get.
        """
        if isinstance(variables, (list, np.ndarray, pd.Index)):  # if iterable
            variables = [v if isinstance(v, str) else str(v) for v in variables]
        else:
            variables = [str(variables)]
        qry: np.ndarray = self.comp.reindex(index=variables).values.flatten()
        if len(qry) == 1:
            qry = qry[0]
        return qry

    def __str__(self):
        """Get a string representation of the composition."""
        s = ""
        if self.name is not None:
            s += self.name + " "
        if self.reservoir is not None:
            s += "Model of " + self.reservoir + " "
        if self.reference is not None:
            s += "from " + self.reference
        s += "."
        return s

    def __repr__(self):
        """Get a string signature of the composition."""
        r = self.__class__.__name__ + "("
        if self.filename is not None:
            r += f"'{Path(self.filename).name}'"
        for par in ["name", "reference", "reservoir"]:
            if getattr(self, par) is not None:
                r += (
                    ",\n"
                    + " " * (len(self.__class__.__name__) + 1)
                    + f"{par}='{getattr(self, par)}'"
                )
        r += ")"
        return r


def all_reference_compositions() -> dict[str, Composition]:
    """
    Get a dictionary of all reference compositions indexed by name.

    Returns
    --------
    dict
    """
    return {r["name"]: Composition(r["composition"]) for ix, r in NORMDB.iterrows()}


def get_reference_composition(name: str) -> Composition:
    """
    Retrieve a particular composition from the reference database.

    Parameters
    ------------
    name : str
        Name of the reference composition model.

    Returns
    --------
    `pyrolite.geochem.norm.Composition`
    """
    res = NORMDB.query(f"name=='{name}'")
    assert len(res) == 1
    res = res.iloc[0]
    name, composition = res["name"], res["composition"]
    return Composition(composition, name=name)


def get_reference_files(
    directory: str | Path | None = None, formats: list[str] | None = None
) -> list[Path]:
    """
    Get a list of the reference composition files.

    Parameters
    -----------
    directory : str, `None`
        Location of reference data files.
    formats : list, `["csv"]`
        List of potential data formats to draw from. Currently only csv will work.

    Returns
    --------
    list
    """
    if formats is None:
        formats = ["csv"]
    directory = directory or (pyrolite_datafolder(subfolder="geochem") / "refcomp")
    directory = Path(directory)
    assert directory.exists() and directory.is_dir()
    files = []
    for fmt in formats:
        files.extend(directory.glob("./*." + fmt))
    return files


def update_database(encoding: str = "cp1252", **kwargs):
    """
    Update the reference composition database.

    Notes
    ------
    This will take all csv files from the geochem/refcomp pyrolite data folder
    and construct a document-based JSON database.
    """
    pd.DataFrame(
        [
            {
                "name": C.name,
                "composition": json.dumps(C._df.query("~value.isnull()").to_dict()),
            }
            for C in [Composition(f, encoding=encoding) for f in get_reference_files()]
        ]
    ).to_json(__dbfile__, indent=4)
