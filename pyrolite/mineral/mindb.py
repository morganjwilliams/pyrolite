import functools
import pandas as pd
import numpy as np
import periodictable as pt
from pathlib import Path
from tinydb import TinyDB, Query
from .transform import formula_to_elemental
from ..util.meta import pyrolite_datafolder
from ..util.database import _list_tindyb_unique_values
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

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
    Parse a composition reference and return the composiiton as a :class:`~pandas.Series`

    Parameters
    -----------
    composition : :class:`str` | :class:`periodictable.formulas.Formula`
    """
    mnrl = None
    if composition in list_minerals():
        mnrl = get_mineral(composition)

    try:  # formulae
        form = pt.formula(composition)
        mnrl = pd.Series(formula_to_elemental(form))
        # could also check for formulae in the database, using f.atoms
    except:
        pass

    assert mnrl is not None
    if drop_zeros:
        mnrl = mnrl[mnrl != 0]
    return mnrl


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
        + [str(a) for a in pt.formula(" ".join(mindf.formula.to_list())).atoms]
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
        db.purge()
        for k, v in mindf.T.to_dict().items():
            db.insert(v)
        db.close()
