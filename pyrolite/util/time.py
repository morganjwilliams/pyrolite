from collections import ChainMap, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .log import Handle
from .meta import pyrolite_datafolder
from .text import titlecase

logger = Handle(__name__)


# get the latest geotimescale data
__data__: Path = max(
    pyrolite_datafolder(subfolder="timescale").glob("geotimescale_*.csv")
)
__colors__: Path = pyrolite_datafolder(subfolder="timescale") / "timecolors.csv"


def listify(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """
    Consdense text information across columns into a single list.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe (or slice of dataframe) to condense along axis.
    axis : int
        Axis to condense along.
    """
    return df.copy(deep=True).apply(list, axis=axis)


def age_name(
    agenamelist: list[str],
    prefixes: list[str] | None = None,
    suffixes: list | None = None,
) -> str:
    """
    Condenses an agename list to a specific agename, given a subset of
    ambiguous_names.

    Parameters
    ----------
    agenamelist : list
        List of name components (i.e. `[Eon, Era, Period, Epoch]`)
    prefixes : list | None
        Name components which occur prior to the higher order classification
        (e.g. `"Upper Triassic"`).
    suffixes : list
        Name components which occur after the higher order classification
        (e.g. `"Cambrian Series 2"`).
    """
    if suffixes is None:
        suffixes: list[str] = ["Stage", "Series"]
    if prefixes is None:
        prefixes: list[str] = ["Lower", "Middle", "Upper"]
    ambiguous_names: list[str] = prefixes + suffixes
    ambig_vars: list[str] = [s.lower().strip() for s in ambiguous_names]
    nameguess: str = agenamelist[-1]
    # Process e.g. Stage 1 => Stage
    nn_nameguess: str = "".join([i for i in nameguess if not i.isdigit()]).strip()

    # check if the name guess corresponds to any of the ambiguous names
    try:
        hit: str | None = next(
            iter(
                [
                    ambiguous_names[ix]
                    for ix, vars in enumerate(ambig_vars)
                    if nn_nameguess.lower().strip() in vars
                ]
            )
        )
    except StopIteration:
        hit = None

    if hit:
        indexstart = len(agenamelist) - 1
        outname = [agenamelist[indexstart]]
        out_index_previous = 0
        ambiguous_name = True
        while ambiguous_name:
            hitphrase = hit
            indexstart -= 1
            nextup = agenamelist[indexstart]
            if hitphrase in prefixes:
                # insert the higher order component after the previous one
                outname.insert(out_index_previous + 1, nextup)
                out_index_previous += 1
            else:
                # insert the higher order component before the previous one
                outname.insert(out_index_previous - 1, nextup)
                out_index_previous -= 1

            _nn_nextupguess = "".join([i for i in nextup if not i.isdigit()]).strip()
            try:
                hit: str | None = next(
                    iter(
                        [
                            ambiguous_names[ix]
                            for ix, vars in enumerate(ambig_vars)
                            if _nn_nextupguess.lower().strip() in vars
                        ]
                    )
                )
            except StopIteration:
                hit = None
            if not hit:
                ambiguous_name = False
        return " ".join(outname)
    else:
        return nameguess


def import_colors(
    filename: Path | str = __colors__, delim: str = "/"
) -> dict[str, tuple[float, float, float, float]]:
    """
    Import a list of timescale names with associated colors.
    """
    c = pd.read_csv(filename).dropna(how="all")
    if delim is not None:  # and ("RGB" in c.columns):
        c["RGB"] = c["RGB"].apply(
            lambda x: tuple(
                [float(i) / 255.0 for i in x.split(delim)] + [1.0]
            )  # add alpha
        )
    return {name: rgb for name, rgb in c.values}


def timescale_reference_frame(
    filename: str | Path | None = None,
    info_cols: list[str] | None = None,
    color_info: dict[str, tuple[float, float, float, float] | str] | None = None,
) -> pd.DataFrame:
    """
    Rearrange the text-based timescale dataframe. Utility function for
    timescale class.

    Parameters
    ----------
    filename : str | pathlib.Path
        File from which to generate the timescale information.
    info_cols : list
        List of columns beyond hierarchial group labels (e.g. Eon, Era..).

    Returns
    -------
    pandas.DataFrame
        Dataframe containing timescale information.
    """
    if filename is None:
        filename = __data__
    if info_cols is None:
        info_cols: list[str] = ["Start", "End", "Aliases"]
    df = pd.read_csv(filename)
    df[["Start", "End"]] = df.loc[:, ["Start", "End"]].apply(
        pd.to_numeric, errors="coerce"
    )
    _df = df.copy(deep=True)
    grps = [i for i in _df.columns if i not in info_cols]
    condensed = _df.loc[:, [i for i in _df.columns if i not in info_cols]].fillna(
        value=""
    )
    _df["Level"] = condensed.apply(
        lambda x: grps[[ix for ix, v in enumerate(x) if v][-1]], axis=1
    )
    condensed = listify(condensed).apply(lambda x: [i for i in x if i])
    _df["Name"] = condensed.apply(age_name)
    _df["Ident"] = condensed.apply("-".join)
    _df["MeanAge"] = _df.apply(lambda x: (x.Start + x.End) / 2, axis=1)
    _df["Unc"] = _df.apply(lambda x: np.abs(x.Start - x.End) / 2, axis=1)

    # Aliases
    _df.Aliases = _df.Aliases.apply(lambda x: [] if pd.isnull(x) else x.split(";"))
    _df.Aliases = _df.apply(lambda x: [x.Name, x.Ident] + x.Aliases, axis=1)
    _df.Aliases = _df.Aliases.apply(lambda x: [i.lower().strip() for i in x])

    colors = color_info or import_colors()
    _df["Color"] = _df.Name.apply(lambda x: colors.get(x, None))
    col_order = (
        ["Ident", "Name", "Level", "Start", "End", "MeanAge", "Unc"]
        + grps
        + ["Aliases", "Color"]
    )

    return _df.loc[:, col_order]


class Timescale:
    def __init__(self, filename: str | Path | None = None):
        """
        Geological Timescale class to provide time-focused utility functions.

        Parameters
        -----------
        filename : `str` | `pathlib.Path`
            Path to the timescale data file.

        Attributes
        ----------
        data : pandas.DataFrame
            Timescale dataframe.
        levels : list
            Hierarchial levels within the timescale.
        """

        self.data: pd.DataFrame = timescale_reference_frame(filename)
        self.levels = [i for i in self.data.Level.unique() if not pd.isnull(i)]
        self.levels: list[str] = [i for i in self.data.columns if i in self.levels]

        def getnan():
            return np.nan, np.nan

        self.locate: defaultdict[str, tuple[float, float]] = defaultdict(getnan)
        self.build()

    def build(self):
        """
        Build the timescale from data within file.
        """
        for ix, g in enumerate(self.levels):
            others = self.levels[ix + 1 :]
            fltr = (
                self.data.loc[:, others].isnull().all(axis=1)
                & ~self.data.loc[:, g].isnull()
            )
            setattr(self, g + "s", self.data.loc[fltr, :])

        dicts = self.data.apply(
            lambda x: {a: (x.Start, x.End) for a in x.Aliases}, axis=1
        )
        # should check that the keys are unique across all of these
        self.locate.update(dict(ChainMap(*dicts)))
        self.data: pd.DataFrame = self.data.set_index("Ident")

    def text2age(
        self, entry: str, nulls: list[str | float | None] | None = None
    ) -> tuple[float, float] | list[tuple[float, float]]:
        """
        Converts a text-based age to the corresponding age range (in Ma).

        String-based entries return (max_age, min_age). Collection-based entries
        return a list of tuples.

        Parameters
        ------------
        entry : str
            String name, or series of string names, for geological age range.

        Returns
        -------
        tuple | list[tuple]
            Tuple or list of tuples.
        """
        if nulls is None:
            nulls: list[None | str | float] = [None, "None", "none", np.nan, "NaN"]
        try:
            entry = float(entry)
            return (entry, entry)
        except ValueError:
            return self.locate[entry.lower().strip()]

    def named_age(self, age: float, level: str = "Specific", **kwargs) -> str | None:
        """
        Converts a numeric age (in Ma) to named age at a specific level.

        Parameters
        ----------
        age : float
            Numeric age in Ma.
        level : str, `{'Supereon', 'Eon', 'Era', 'Period', 'Superepoch', 'Epoch', 'Age', 'Specific'}`
            Level of specificity.

        Returns
        -------
        str
            String representation for the entry.
        """

        level = titlecase(level)

        def wthn_rng(x):
            return (age <= x.Start) & (age >= x.End)

        relevant = self.data.loc[self.data.apply(wthn_rng, axis=1).values, :]
        if level == "Specific":  # take the rightmost grouping
            relevant = relevant.loc[:, self.levels]
            counts = (~pd.isnull(relevant)).count(axis=1)
            if sum(counts == counts.max()) > 1:
                idx_rel_row = counts.index[
                    max([ix for (ix, r) in enumerate(counts) if r == counts.iloc[0]])
                ]
            else:
                idx_rel_row = counts.idxmax()
            rel_row = relevant.loc[idx_rel_row, :]
            return age_name(rel_row[~pd.isnull(rel_row)].to_list(), **kwargs)
        else:
            try:
                unique_values = relevant.loc[:, level].unique()
                return unique_values[~pd.isnull(unique_values)][0]
            except IndexError:
                # likely no relevant level name.
                logger.debug(f"No name found at level {level} for age {age} Ma.")
