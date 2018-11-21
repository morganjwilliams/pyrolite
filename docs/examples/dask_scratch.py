import dask

# requires adding dask, graphviz to req

import dask.dataframe as dd
import pandas as pd

from pyrolite.util.pd import test_df



@dask.delayed
def column_ordered_append(df1, df2, **kwargs):
    """
    Appends one dataframe to another, preserving the column order of the
    first and appending new columns on the right. Also accepts and passes on
    standard keyword arguments for pd.DataFrame.append.

    Parameters
    ------------
    df1: pd.DataFrame
        The dataframe for which columns order is preserved in the output.
    df2: pd.DataFrame
        The dataframe for which new columns are appended to the output.

    """
    outcols = list(df1.columns) + [i for i in df2.columns
                                   if not i in df1.columns]
    return df1.append(df2,  **kwargs).reindex(columns=outcols)

@dask.delayed
def accumulate(dfs, ignore_index=False):
    """
    Accumulate an iterable containing pandas dataframes to a single frame.
    """
    acc = None
    for df in dfs:
        if acc is None:
            acc = df
        else:
            acc = column_ordered_append(acc, df, ignore_index=ignore_index)
    return acc

df0 = test_df()
others = [test_df()]*4
result1 = accumulate([df0]+others)
result2 = accumulate([df0]+others)
result2
result3 = accumulate([result1, result2])
#result3.compute().compute()
result3.visualize()

@dask.delayed
def inc(x):
    return x + 1

@dask.delayed
def double(x):
    return x + 2

@dask.delayed
def add(x, y):
    return x + y

data = [1, 2, 3, 4, 5]

output = []
for x in data:
    a = inc(x)
    b = double(x)
    c = add(a, b)
    output.append(c)

total = dask.delayed(sum)(output)

total.compute()
