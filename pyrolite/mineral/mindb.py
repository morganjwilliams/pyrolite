"""
Submodule for accessing the rock forming mineral database.

Notes
-----
Accessing and modifying the database across multiple with multiple threads/processes
*could* result in database corruption (e.g. through repeated truncation etc).
"""
import functools
import pandas as pd
import numpy as np
import periodictable as pt
from pathlib import Path
from tinydb import TinyDB, Query
from .transform import formula_to_elemental, merge_formulae
from ..util.meta import pyrolite_datafolder
from ..util.database import _list_tindyb_unique_values
from ..util.log import Handle

logger = Handle(__name__)

__dbpath__ = pyrolite_datafolder(subfolder="mineral") / "mindb.json"


@functools.lru_cache(maxsize=None)  # cache outputs for speed
def list_groups():
    """
    List the mineral groups present in the mineral database.

    Returns
    ----------
    :class:`list`
    """
    return _list_tindyb_unique_values("group", dbpath=__dbpath__)


@functools.lru_cache(maxsize=None)  # cache outputs for speed
def list_minerals():
    """
    List the minerals present in the mineral database.

    Returns
    ----------
    :class:`list`
    """
    return _list_tindyb_unique_values("name", dbpath=__dbpath__)


@functools.lru_cache(maxsize=None)  # cache outputs for speed
def list_formulae():
    """
    List the mineral formulae present in the mineral database.

    Returns
    ----------
    :class:`list`
    """
    return _list_tindyb_unique_values("formula", dbpath=__dbpath__)


def get_mineral(name="", dbpath=None):
    """
    Get a specific mineral from the database.

    Parameters
    ------------
    name : :class:`str`
        Name of the desired mineral.
    dbpath : :class:`pathlib.Path`, :class:`str`
        Optional overriding of the default database path.

    Returns
    --------
    :class:`pd.Series`
    """
    if dbpath is None:
        dbpath = __dbpath__

    assert name in list_minerals()
    with TinyDB(str(dbpath)) as db:
        out = db.get(Query().name == name)

    return pd.Series(out)


def parse_composition(composition, drop_zeros=True):
    """
    Parse a composition reference to provide an ionic elemental version in the form of a
    :class:`~pandas.Series`. Currently accepts :class:`pandas.Series`,
    :class:`periodictable.formulas.Formula`
    and structures which will directly convert to :class:`pandas.Series`
    (list of tuples, dict).

    Parameters
    -----------
    composition : :class:`str` | :class:`periodictable.formulas.Formula` | :class:`pandas.Series`
        Name of a mineral, a formula or composition as a series
    drop_zeros : :class:`bool`
        Whether to drop compositional zeros.

    Returns
    --------
    mineral : :class:`pandas.Series`
        Composition formatted as a series.
    """
    mineral = None
    if composition is not None:
        if isinstance(composition, pd.Series):
            # convert to molecular oxides, then to formula, then to wt% elemental
            components = [pt.formula(c) for c in composition.index]
            values = composition.values
            formula = merge_formulae(
                [v / c.mass * c for v, c in zip(values, components)]
            )
            mineral = pd.Series(formula_to_elemental(formula))
        elif isinstance(composition, pt.formulas.Formula):
            mineral = pd.Series(formula_to_elemental(composition))
        elif isinstance(composition, str):
            if composition in list_minerals():
                mineral = get_mineral(composition)
            else:
                try:  # formulae
                    form = pt.formula(composition)
                    mineral = pd.Series(formula_to_elemental(form))
                    # could also check for formulae in the database, using f.atoms
                except:
                    pass
        else:
            mineral = parse_composition(pd.Series(composition))

    if drop_zeros and mineral is not None:
        mineral = mineral[mineral != 0]
    return mineral


def get_mineral_group(group=""):
    """
    Extract a mineral group from the database.

    Parameters
    -----------
    group : :class:`str`
        Group to extract from the mineral database.

    Returns
    ---------
    :class:`pandas.DataFrame`
        Dataframe of group members and compositions.
    """
    assert group in list_groups()
    with TinyDB(str(__dbpath__)) as db:
        grp = db.search(Query().group == group)

    df = pd.DataFrame(grp)
    meta, chem = (
        ["name", "formula"],
        [i for i in df.columns if i not in ["name", "formula", "group"]],
    )
    df = df.reindex(columns=meta + chem)
    df.loc[:, chem] = df.loc[:, chem].apply(pd.to_numeric)
    df = df.loc[:, (df != 0).any(axis=0)]  # remove zero-only columns
    return df


def update_database(path=None, **kwargs):
    """
    Update the mineral composition database.

    Parameters
    -----------
    path : :class:`str` | :class:`pathlib.Path`
        The desired filepath for the JSON database.

    Notes
    ------
    This will take the 'mins.csv' file from the mineral pyrolite data folder
    and construct a document-based JSON database.
    """
    mindf = pd.read_csv(pyrolite_datafolder(subfolder="mineral") / "mins.csv")
    mindf = mindf.reindex(
        columns=mindf.columns.tolist()
        + [str(a) for a in pt.formula(" ".join(list(mindf.formula.values))).atoms]
    )
    for ix in mindf.index:  # add elemental compositions
        el = parse_composition(pt.formula(mindf.loc[ix, "formula"]))
        mindf.loc[ix, el.index] = el

    mindf = mindf.fillna(0.0)

    if path is None:
        path = __dbpath__

    path = Path(path).with_suffix(".json")

    # name group formula composition
    with TinyDB(str(path)) as db:
        db.truncate()
        for k, v in mindf.T.to_dict().items():
            db.insert(v)
        db.close()
