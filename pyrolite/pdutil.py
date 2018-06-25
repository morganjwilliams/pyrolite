import pandas as pd


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
