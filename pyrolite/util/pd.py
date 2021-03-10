import pandas as pd
import hashlib
from pathlib import Path
import numpy as np
from .meta import subkwargs
from .log import Handle

logger = Handle(__name__)


def drop_where_all_empty(df):
    """
    Drop rows and columns which are completely empty.

    Parameters
    ----------
    df : :class:`pandas.DataFrame` | :class:`pandas.Series`
        Pandas object to ensure is in the form of a series.
    """
    for ix in range(len(df.axes)):
        df = df.dropna(how="all", axis=ix)
    return df


def read_table(filepath, index_col=0, **kwargs):
    """
    Read tabluar data from an excel or csv text-based file.

    Parameters
    ------------
    filepath : :class:`str` | :class:`pathlib.Path`
        Path to file.

    Returns
    --------
    :class:`pandas.DataFrame`
    """
    filepath = Path(filepath)
    ext = filepath.suffix.replace(".", "")
    assert ext in ["xls", "xlsx", "csv"]
    if ext in ["xls", "xlsx"]:
        reader, kw = pd.read_excel, dict(engine="openpyxl")
    elif ext in ["csv"]:
        reader, kw = pd.read_csv, {}
    else:
        raise NotImplementedError("Only .xls* and .csv currently supported.")
    df = reader(
        str(filepath), index_col=index_col, **subkwargs({**kw, **kwargs}, reader)
    )
    df = drop_where_all_empty(df)
    return df


def column_ordered_append(df1, df2, **kwargs):
    """
    Appends one dataframe to another, preserving the column order of the
    first and adding new columns on the right. Also accepts and passes on
    standard keyword arguments for pd.DataFrame.append.

    Parameters
    ------------
    df1 : :class:`pandas.DataFrame`
        The dataframe for which columns order is preserved in the output.
    df2 : :class:`pandas.DataFrame`
        The dataframe for which new columns are appended to the output.

    Returns
    --------
    :class:`pandas.DataFrame`
    """
    outcols = list(df1.columns) + [i for i in df2.columns if not i in df1.columns]
    return df1.append(df2, sort=False, **kwargs).reindex(columns=outcols)


def accumulate(dfs, ignore_index=False, trace_source=False, names=[]):
    """
    Accumulate an iterable containing multiple :class:`pandas.DataFrame` to a single
    frame.

    Parameters
    -----------
    dfs : :class:`list`
        Sequence of dataframes.
    ignore_index : :class:`bool`
        Whether to ignore the indexes upon joining.
    trace_source : :class:`bool`
        Whether to retain a reference to the source of the data rows.
    names : :class:`list`
        Names to use in place of indexes for source names.

    Returns
    --------
    :class:`pandas.DataFrame`
        Accumulated dataframe.
    """
    acc = None
    for ix, df in enumerate(dfs):
        if trace_source:
            if names:
                df["src_idx"] = names[ix]
            else:
                df["src_idx"] = ix
        if acc is None:
            acc = df
        else:
            acc = column_ordered_append(acc, df, ignore_index=ignore_index)
    return acc


def to_frame(ser):
    """
    Simple utility for converting to :class:`pandas.DataFrame`.

    Parameters
    ----------
    ser : :class:`pandas.Series` | :class:`pandas.DataFrame`
        Pandas object to ensure is in the form of a dataframe.

    Returns
    --------
    :class:`pandas.DataFrame`
    """

    if isinstance(ser, pd.Series):  # using series instead of dataframe
        df = ser.to_frame().T
    elif isinstance(ser, pd.DataFrame):  # 1 column slice
        if ser.columns.size == 1:
            df = ser.T
        else:
            df = ser
    else:
        msg = "Conversion from {} to dataframe not yet implemented".format(type(ser))
        raise NotImplementedError(msg)

    return df


def to_ser(df):
    """
    Simple utility for converting single column :class:`pandas.DataFrame`
    to :class:`pandas.Series`.

    Parameters
    ----------
    df : :class:`pandas.DataFrame` | :class:`pandas.Series`
        Pandas object to ensure is in the form of a series.

    Returns
    --------
    :class:`pandas.Series`
    """
    if isinstance(df, pd.Series):  # passed series instead of dataframe
        ser = df
    elif isinstance(df, pd.DataFrame):
        assert (df.columns.size == 1) or (
            df.index.size == 1
        ), """Can't convert DataFrame to Series:
              either columns or index need to have size 1."""
        if df.columns.size == 1:
            ser = df.iloc[:, 0]
        else:
            ser = df.iloc[0, :]
    else:
        msg = "Conversion from {} to series not yet implemented".format(type(df))
        raise NotImplementedError(msg)

    return ser


def to_numeric(df, errors: str = "coerce", exclude=["float", "int"]):
    """
    Converts non-numeric columns to numeric type where possible.

    Notes
    -----

    Avoid using .loc or .iloc on the LHS to make sure that data dtypes
    are propagated.
    """
    cols = df.select_dtypes(exclude=exclude).columns
    df[cols] = df.loc[:, cols].apply(pd.to_numeric, errors=errors)
    return df


def zero_to_nan(df, rtol=1e-5, atol=1e-8):
    """
    Replace floats close, less or equal to zero with np.nan in a dataframe.

    Parameters
    ------------
    df : :class:`pandas.DataFrame`
        DataFrame to censor.
    rtol : :class:`float`
        The relative tolerance parameter.
    atol : :class:`float`
        The absolute  tolerance parameter.

    Returns
    --------
    :class:`pandas.DataFrame`
        Censored DataFrame.
    """
    cols = [
        name
        for (name, type) in zip(df.columns, df.dtypes)
        if isinstance(type, np.float)
    ]
    df.loc[:, cols] = np.where(
        np.isclose(df[cols].values, 0.0, rtol=rtol, atol=atol), np.nan, df[cols].values
    )
    df.loc[:, cols] = np.where(df[cols].values < 0.0, np.nan, df[cols].values)
    return df


def outliers(
    df,
    cols=[],
    detect=lambda x, quantile, qntls: (
        (x > quantile.loc[qntls[0], x.name]) & (x < quantile.loc[qntls[1], x.name])
    ),
    quantile_select=(0.02, 0.98),
    logquantile=False,
    exclude=False,
):
    """"""
    if not cols:
        cols = df.columns
    colfltr = (df.dtypes == np.float) & ([i in cols for i in df.columns])
    low, high = np.min(quantile_select), np.max(quantile_select)
    if not logquantile:
        quantile = df.loc[:, colfltr].quantile([low, high])
    else:
        quantile = df.loc[:, colfltr].apply(np.log).quantile([low, high])
    whereout = (
        df.loc[:, colfltr]
        .apply(detect, args=(quantile, quantile_select), axis=0)
        .sum(axis=1)
        > 0
    )
    if not exclude:
        whereout = np.logical_not(whereout)
    return df.loc[whereout, colfltr]


def concat_columns(df, columns=None, astype=str, **kwargs):
    """
    Concatenate strings across columns.

    Parameters
    -----------
    df : :class:`pandas.DataFrame`
        Dataframe to concatenate.
    columns : :class:`list`
        List of columns to concatenate.
    astype : :class:`type`
        Type to convert final concatenation to.

    Returns
    -------
    :class:`pandas.Series`
    """
    if columns is None:
        columns = df.columns
    out = pd.Series(index=df.index, **kwargs)
    for ix, c in enumerate(columns):
        if ix == 0:
            out = df.loc[:, c].astype(astype)
        else:
            out += df.loc[:, c].astype(astype)
    return out


def uniques_from_concat(df, columns=None, hashit=True):
    """
    Creates ideally unique keys from multiple columns.
    Optionally hashes string to standardise length of identifier.

    Parameters
    ------------
    df : :class:`pandas.DataFrame`
        DataFrame to create indexes for.
    columns : :class:`list`
        Columns to use in the string concatenatation.
    hashit : :class:`bool`, :code:`True`
        Whether to use a hashing algorithm to create the key from a typically
        longer string.

    Returns
    ---------
    :class:`pandas.Series`
    """
    if columns is None:
        columns = df.columns

    if hashit:

        def fmt(ser):
            ser = ser.str.encode("UTF-8")
            ser = ser.apply(lambda x: hashlib.md5(x).hexdigest())
            return ser

    else:
        fmt = lambda x: x.str.encode("UTF-8")

    return fmt(concat_columns(df, columns, dtype="category"))


def df_from_csvs(csvs, dropna=True, ignore_index=False, **kwargs):
    """
    Takes a list of .csv filenames and converts to a single DataFrame.
    Combines columns across dataframes, preserving order of the first entered.

    E.g.
    SiO2, Al2O3, MgO, MnO, CaO
    SiO2, MgO, FeO, CaO
    SiO2, Na2O, Al2O3, FeO, CaO
    =>
    SiO2, Na2O, Al2O3, MgO, FeO, MnO, CaO
    - Existing neighbours take priority (i.e. FeO won't be inserted bf Al2O3)
    - Earlier inputs take priority (where ordering is ambiguous, place the earlier first)

    Todo
    ----
    Attempt to preserve column ordering across column sets, assuming
    they are generally in the same order but preserving only some of the
    information.
    """
    cols = []
    dfs = []
    for t in csvs:
        dfs.append(pd.read_csv(t, **kwargs))
        cols = cols + [i for i in dfs[-1].columns if i not in cols]

    df = accumulate(dfs, ignore_index=ignore_index)
    return df
