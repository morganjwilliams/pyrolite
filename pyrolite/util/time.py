import pandas as pd
import numpy as np
from .meta import pyrolite_datafolder
from .text import titlecase
from collections import ChainMap, defaultdict
from .log import Handle

logger = Handle(__name__)


__data__ = pyrolite_datafolder(subfolder="timescale") / "geotimescale_202003.csv"
__colors__ = pyrolite_datafolder(subfolder="timescale") / "timecolors.csv"


def listify(df, axis=1):
    """
    Consdense text information across columns into a single list.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe (or slice of dataframe) to condense along axis.
    axis : :class:`int`
        Axis to condense along.
    """
    return df.copy(deep=True).apply(list, axis=axis)


def age_name(
    agenamelist, prefixes=["Lower", "Middle", "Upper"], suffixes=["Stage", "Series"]
):
    """
    Condenses an agename list to a specific agename, given a subset of
    ambiguous_names.

    Parameters
    ----------
    agenamelist : :class:`list`
        List of name components (i.e. :code:`[Eon, Era, Period, Epoch]`)
    prefixes : :class:`list`
        Name components which occur prior to the higher order classification
        (e.g. :code:`"Upper Triassic"`).
    suffixes : :class:`list`
        Name components which occur after the higher order classification
        (e.g. :code:`"Cambrian Series 2"`).
    """
    ambiguous_names = prefixes + suffixes
    ambig_vars = [s.lower().strip() for s in ambiguous_names]
    nameguess = agenamelist[-1]
    # Process e.g. Stage 1 => Stage
    nn_nameguess = "".join([i for i in nameguess if not i.isdigit()]).strip()

    # check if the name guess corresponds to any of the ambiguous names
    hit = [
        ambiguous_names[ix]
        for ix, vars in enumerate(ambig_vars)
        if nn_nameguess.lower().strip() in vars
    ][0:1]

    if hit:
        indexstart = len(agenamelist) - 1
        outname = [agenamelist[indexstart]]
        out_index_previous = 0
        ambiguous_name = True
        while ambiguous_name:
            hitphrase = hit[0]
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
            hit = [
                ambiguous_names[ix]
                for ix, vars in enumerate(ambig_vars)
                if _nn_nextupguess.lower().strip() in vars
            ][0:1]
            if not hit:
                ambiguous_name = False
        return " ".join(outname)
    else:
        return nameguess


def import_colors(filename=__colors__, delim="/"):
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
    filename=__data__, info_cols=["Start", "End", "Aliases"], color_info=None
):
    """
    Rearrange the text-based timescale dataframe. Utility function for
    timescale class.

    Parameters
    ----------
    filename : :class:`str` | :class:`pathlib.Path`
        File from which to generate the timescale information.
    info_cols : :class:`list`
        List of columns beyond hierarchial group labels (e.g. Eon, Era..).

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing timescale information.
    """

    df = pd.read_csv(filename)
    df[["Start", "End"]] = df.loc[:, ["Start", "End"]].apply(
        pd.to_numeric, errors="coerce"
    )
    _df = df.copy(deep=True)
    grps = [i for i in _df.columns if not i in info_cols]
    condensed = _df.loc[:, [i for i in _df.columns if not i in info_cols]].fillna(
        value=""
    )
    _df["Level"] = condensed.apply(
        lambda x: grps[[ix for ix, v in enumerate(x) if v][-1]], axis=1
    )
    condensed = listify(condensed).apply(lambda x: [i for i in x if i])
    _df["Name"] = condensed.apply(age_name)
    _df["Ident"] = condensed.apply("-".join)
    _df["MeanAge"] = _df.apply(lambda x: (x.Start + x.End) / 2, axis=1)
    _df["Unc"] = _df.apply(lambda x: np.abs((x.Start - x.End)) / 2, axis=1)

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


class Timescale(object):
    def __init__(self, filename=None):
        """
        Geological Timescale class to provide time-focused utility functions.

        Parameters
        -----------
        filename : :class:`str` | :class:`pathlib.Path`
            Path to the timescale data file.

        Attributes
        ----------
        data : :class:`pandas.DataFrame`
            Timescale dataframe.
        levels : :class:`list`
            Hierarchial levels within the timescale.
        """
        if filename is None:
            self.data = timescale_reference_frame()
        else:
            self.data = timescale_reference_frame(filename)
        self.levels = [i for i in self.data.Level.unique() if not pd.isnull(i)]
        self.levels = [i for i in self.data.columns if i in self.levels]

        def getnan():
            return np.nan, np.nan

        self.locate = defaultdict(getnan)
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
        self.data = self.data.set_index("Ident")

    def text2age(self, entry, nulls=[None, "None", "none", np.nan, "NaN"]):
        """
        Converts a text-based age to the corresponding age range (in Ma).

        String-based entries return (max_age, min_age). Collection-based entries
        return a list of tuples.

        Parameters
        ------------
        entry : :class:`str`
            String name, or series of string names, for geological age range.

        Returns
        -------
        :class:`tuple` | :class:`list` (:class:`tuple`)
            Tuple or list of tuples.
        """
        try:
            entry = np.float(entry)
            return (entry, entry)
        except ValueError:
            return self.locate[entry.lower().strip()]

    def named_age(self, age, level="Specific", **kwargs):
        """
        Converts a numeric age (in Ma) to named age at a specific level.

        Parameters
        ----------
        age : :class:`float`
            Numeric age in Ma.
        level : :class:`str`, :code:`{'Eon', 'Era', 'Period', 'Superepoch', 'Epoch', 'Age', 'Specific'}`
            Level of specificity.

        Returns
        -------
        :class:`str`
            String representation for the entry.
        """

        level = titlecase(level)
        wthn_rng = lambda x: (age <= x.Start) & (age >= x.End)
        relevant = self.data.loc[self.data.apply(wthn_rng, axis=1).values, :]
        if level == "Specific":  # take the rightmost grouping
            relevant = relevant.loc[:, self.levels]
            counts = (~pd.isnull(relevant)).count(axis=1)
            if sum(counts == counts.max()) > 1:
                idx_rel_row = counts.index[
                    max([ix for (ix, r) in enumerate(counts) if r == counts[0]])
                ]
            else:
                idx_rel_row = counts.idxmax()
            rel_row = relevant.loc[idx_rel_row, :]
            return age_name(rel_row[~pd.isnull(rel_row)], **kwargs)
        else:
            unique_values = relevant.loc[:, level].unique()
            return unique_values[~pd.isnull(unique_values)][0]
