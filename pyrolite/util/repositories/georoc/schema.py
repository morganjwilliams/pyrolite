import logging
import re
import pandas as pd
import numpy as np
from ...text import titlecase, split_records
from ....geochem.parse import tochem, check_multiple_cation_inclusion
from ....geochem.transform import aggregate_cation
from ....geochem.norm import scale_multiplier
from .parse import parse_citations, parse_values, parse_DOI

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def format_GEOROC_response(content: str, start_chem="SiO2", end_chem="Nd143Nd144"):
    """
    Formats decoded content from GEOROC as a :class:`pandas.DataFrame`

    Parameters
    ---------
    content : :class:`str`
        Decoded string from GEOROC response.

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    # GEOROC Specific Data Working
    data, ref = re.split("\s?References:\s+", content)
    datalines = [re.split(r'"\s?,\s?"', line) for line in re.split(r",\r", data)]
    cols = [i.replace('"', "").replace(",", "") for i in datalines[0]]
    cols = [titlecase(h, abbrv=["ID"]) for h in cols]
    start = 1
    finish = len(datalines)
    if datalines[-1][0].strip().startswith("Abbreviations"):
        finish -= 1
    df = pd.DataFrame(datalines[start:finish], columns=cols)
    cols = list(df.columns)
    df = df.applymap(lambda x: str(x).replace('"', ""))

    # Location names are extended with newlines
    df.Location = df.Location.apply(lambda x: str(x).replace("\r\n", " / "))

    df.Citations = df.Citations.apply(lambda x: re.findall(r"[\d]+", x))
    # df = df.drop(index=df.index[~df.Citations.apply(lambda x: len(x))])
    # Drop Empty Rows
    df = df.dropna(how="all", axis=0)
    df = df.set_index("UniqueID", drop=True)
    df = df.apply(parse_values, axis=1)

    # Translate headers and data units
    cols = tochem([c.replace("(wt%)", "").replace("(ppm)", "") for c in df.columns])
    start = cols.index("SiO2")
    end = cols.index("143Nd144Nd")
    where_ppm = [
        (("ppm" in t) and (ix >= start and ix <= end))
        for ix, t in enumerate(df.columns)
    ]

    # Rename columns
    df.columns = cols
    headercols = list(df.columns[:start])
    chemcols = list(df.columns[start:end])
    trailingcols = list(df.columns[end:])  # trailing are generally isotope ratios
    # Numeric data

    numheaders = [
        "ElevationMin",
        "ElevationMax",
        "LatitudeMin",
        "LatitudeMax",
        "LongitudeMin",
        "LongitudeMax",
        "Min.Age(yrs.)",
        "Max.Age(yrs.)",
    ]

    numeric_cols = numheaders + chemcols + trailingcols
    # can include duplicates at this stage.
    numeric_cols = [i for i in df.columns if i in numeric_cols]
    numeric_ixs = [ix for ix, i in enumerate(df.columns) if i in numeric_cols]
    df[numeric_cols] = df.iloc[:, numeric_ixs].apply(
        pd.to_numeric, errors="coerce", axis=1
    )
    # remove <0.
    chem_ixs = [ix for ix, i in enumerate(df.columns) if i in chemcols]
    df.iloc[:, chem_ixs] = df.iloc[:, chem_ixs].mask(
        df.iloc[:, chem_ixs] <= 0.0, other=np.nan
    )

    # units conversion -- convert to Wt%
    df.iloc[:, where_ppm] *= scale_multiplier("ppm", "Wt%")

    # deal with duplicate columns
    collist = list(df.columns)
    dup_chemcols = df.columns[
        df.columns.duplicated() & [i in chemcols for i in collist]
    ]
    for chem in dup_chemcols:
        # replace the first (non-duplicated) column with the sum
        ix = collist.index(chem)
        df.iloc[:, ix] = df.loc[:, chem].apply(np.nansum, axis=1)

    df = df.iloc[:, ~df.columns.duplicated()]

    # Process the reference data.
    reflines = split_records(ref)
    reflines = [line.replace('"', "") for line in reflines]
    reflines = [line.replace("\r\n", "") for line in reflines]
    reflines = [parse_citations(i) for i in reflines if i]
    refdf = pd.DataFrame.from_records(reflines).set_index("key", drop=True)
    # Replace the reference indexes with references.
    df.Citations = df.Citations.apply(
        lambda lst: "; ".join([refdf.loc[x, "value"] for x in lst])
    )
    df["doi"] = df.Citations.apply(parse_DOI)
    return df


def georoc_munge(df):
    """
    Collection of munging and feature adding functions for GEROROC data.

    Todo
    ------
        * combine GEOL and AGE columns for geological ages
    """
    df = aggregate_cation(df, "Ti", form="element")
    df.loc[:, "GeolAge"] = df.loc[:, "Geol."].replace("None", "") + df.Age

    df.loc[:, "Lat"] = (df.LatitudeMax + df.LatitudeMin) / 2.0
    df.loc[:, "Long"] = (df.LongitudeMax + df.LongitudeMin) / 2.0
    return df
