from types import MethodType
import pandas as pd
import hashlib

def column_ordered_append(df1, df2, **kwargs):
    outcols = list(df1.columns) + [i for i in df2.columns
                                   if not i in df1.columns]
    return df1.append(df2,  **kwargs).reindex(columns=outcols)


def accumulate(dfs):
    acc = None
    for df in dfs:
        if acc is None:
            acc = df
        else:
            acc = column_ordered_append(acc, df, ignore_index=False)
    return acc


def to_frame(df):
    """
    Simple utility for converting to pandas dataframes.
    """

    if type(df) == pd.Series:  # using series instead of dataframe
        df = df.to_frame().T
    elif type(df) == pd.DataFrame:  # 1 column slice
        if df.columns.size == 1:
            df = df.T

    return df


def to_numeric(df: pd.DataFrame,
               exclude: list = [],
               errors: str = 'coerce'):
    """
    Takes all non-metadata columns and converts to numeric type where possible.

    Could be reimplemented to operate per-column for better memory performance.
    """
    num_headers = tuple([i for i in df.columns if i not in exclude])
    df.loc[:, num_headers] = df.loc[:, num_headers].apply(pd.to_numeric,
                                                          axis=0,
                                                          errors=errors)
    return df


def concat_columns(df, columns, astype=str, **kwargs):
    out = pd.Series(index=df.index, **kwargs)
    for ix, c in enumerate(columns):
        if ix == 0:
            out = df.loc[:, c].astype(astype)
        else:
            out += df.loc[:, c].astype(astype)
    return out


def uniques_from_concat(df, cols, hashit=True):
    """
    Creates ideally unique keys from multiple columns.
    Optionally hashes string to standardise length of identifier.
    """
    if hashit:
        fmt = lambda x: hashlib.md5(x.encode('UTF-8')).hexdigest()
    else:
        fmt = lambda x: x.encode('UTF-8')

    return concat_columns(df, cols, dtype='category').apply(fmt)


def df_from_csvs(csvs, dropna=True, **kwargs):
    """
    Takes a list of .csv filenames and converts to a single DataFrame.
    Combines columns across dataframes, preserving order of the first entered.

    TODO: Attempt to preserve column ordering across column sets, assuming
    they are generally in the same order but preserving only some of the
    information.

    E.g.
    SiO2, Al2O3, MgO, MnO, CaO
    SiO2, MgO, FeO, CaO
    SiO2, Na2O, Al2O3, FeO, CaO
    =>
    SiO2, Na2O, Al2O3, MgO, FeO, MnO, CaO
    - Existing neighbours take priority (i.e. FeO won't be inserted bf Al2O3)
    - Earlier inputs take priority (where ordering is ambiguous, place the earlier first)
    """
    cols = []
    dfs = []
    for ix, t in enumerate(csvs):
        print(f'Reading {t}')
        dfs.append(pd.read_csv(t, **kwargs))
        cols = cols + [i for i in dfs[-1].columns if i not in cols]

    DF = pd.DataFrame(columns=cols)  # Initialise DataFrame
    DF = pd.concat([DF] + dfs, axis=0).reset_index(drop=True)  # Reset Index
    del dfs
    DF = DF.reindex(columns=cols)  # Reset Columns
    if dropna:
        DF = DF.dropna('columns', how='all')
    return DF


def pickle_from_csvs(targets, out_filename, sep='\t', suffix='.pkl'):
    df = df_from_csvs(targets, sep=sep, low_memory=False)
    sparse_pickle_df(df, out_filename, suffix=suffix)


def sparse_pickle_df(df: pd.DataFrame, filename, suffix='.pkl'):
    """
    Converts dataframe to sparse dataframe before pickling to disk.
    """
    df.to_sparse().to_pickle(filename.with_suffix(suffix))


def load_sparse_pickle_df(filename, suffix='.pkl', keep_sparse=False):
    """
    Loads sparse dataframe from disk, with optional densification.
    """
    if keep_sparse:
        return pd.read_pickle(filename.with_suffix(suffix))
    else:
        return pd.read_pickle(filename.with_suffix(suffix)).to_dense()
