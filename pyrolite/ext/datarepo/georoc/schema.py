import logging
import re
import pandas as pd
import numpy as np
from ....util.text import titlecase, split_records
from ....geochem.parse import check_multiple_cation_inclusion
from ....geochem.transform import aggregate_element
from ....util.units import scale
from ....geochem.ind import __common_elements__, __common_oxides__
from ....geochem.validate import is_isotoperatio
from .parse import parse_citations, parse_values, parse_DOI, columns_to_namesunits

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def parse_GEOROC_response(content: str):
    """
    Formats decoded content from GEOROC as a :class:`pandas.DataFrame`

    Parameters
    ---------
    content : :class:`str`
        Decoded string from GEOROC response.

    Returns
    -------
    :class:`pandas.DataFrame`

    Notes
    -----
        * Chemical abundance data are output as Wt% by default.

    Todo
    ------
        * Use custom types as column headers to avoid needing tuples
    """
    # parse the data and references separately
    data, ref = re.split("\s?References:\s+", content)
    datadf = format_GEOROC_table(data)
    refdf = format_GEOROC_references(ref)
    names, units = [i[0] for i in datadf.columns], [i[1] for i in datadf.columns]
    datadf.index.name = datadf.index.name[0]
    datadf.columns = names
    duplicated = list(datadf.columns[datadf.columns.duplicated()])
    if duplicated:  # could deal with duplicated columns here
        logger.warning("Duplicated columns: {}".format(", ".join(duplicated)))

    # Replace the reference indexes with references.
    datadf.Citations = datadf.Citations.apply(
        lambda lst: "; ".join([refdf.loc[x, "value"] for x in lst])
    )
    datadf.loc[:, "doi"] = datadf.Citations.apply(parse_DOI)
    return datadf


def format_GEOROC_table(content):
    """
    Parameters
    ---------
    content : :class:`str`
        Decoded string from GEOROC response.

    Returns
    -------
    :class:`pandas.DataFrame`

    Notes
    -----
        * Chemical abundance data are output as Wt% by default.

    Todo
    ------
        * Use custom types as column headers to avoid needing tuples
    """
    datalines = [re.split(r'"\s?,\s?"', line) for line in re.split(r",\r", content)]

    logger.debug("Translating Columns")
    cols = [i.replace('"', "").replace(",", "") for i in datalines[0]]
    translate = {
        c: {"fmt": f, "units": u} for c, f, u in zip(cols, *columns_to_namesunits(cols))
    }
    # use composite tuple (name, units) headers to ensure unique column names
    cols = [(translate[c]["fmt"], translate[c]["units"]) for c in cols]
    ppm_columns = [c for c in cols if c[1] == "ppm"]

    logger.debug("Constructing DataFrame")
    finish = len(datalines)
    if datalines[-1][0].strip().startswith("Abbreviations"):
        finish -= 1
    df = pd.DataFrame(datalines[1:finish], columns=cols)

    duplicated = list(df.columns[df.columns.duplicated()])
    if duplicated:
        msg = "Duplicate columns detected: {}".format(duplicated)
        logger.warning(msg)
    logger.debug("Cleaning DataFrame")
    df = df.applymap(lambda x: str(x).replace('"', ""))  # remove extraneous "
    # Location names are extended with newlines
    df[("Location", None)] = df[("Location", None)].apply(
        lambda x: str(x).replace("\r\n", " / ")
    )
    # convert citations string to list of citation #s
    df[("Citations", None)] = df[("Citations", None)].apply(
        lambda x: re.findall(r"[\d]+", x)
    )

    logger.debug("Dropping Empty Rows")
    df = df.dropna(how="all", axis=0)
    logger.debug("Reindexing")
    df = df.set_index(("UniqueID", None), drop=True)
    logger.debug("Parsing Data")
    df = df.apply(parse_values, axis=1)

    chems = __common_oxides__ | __common_elements__
    chemcols = [i for i in df.columns if i[0] in chems]
    isocols = [c for c in df.columns if is_isotoperatio(c[0])]  # isotope ratios
    iniisocols = [
        c
        for c in df.columns
        if is_isotoperatio(c[0].replace("Ini", "")) and (not is_isotoperatio(c[0]))
    ]  # initial isotope ratios

    numheaders = [
        c
        for c in df.columns
        if c[0]
        in [
            "ElevationMin",
            "ElevationMax",
            "LatitudeMin",
            "LatitudeMax",
            "LongitudeMin",
            "LongitudeMax",
            "MinAge",
            "MaxAge",
        ]
    ]

    numeric_cols = numheaders + chemcols + isocols + iniisocols
    logger.debug("Converting numeric data for columns : {}".format(numeric_cols))
    df[numeric_cols] = df.loc[:, numeric_cols].apply(
        pd.to_numeric, errors="coerce", axis=1
    )
    # remove <0.
    df.loc[:, chemcols] = df.loc[:, chemcols].mask(
        df.loc[:, chemcols] <= 0.0, other=np.nan
    )
    # units conversion -- convert to Wt%
    logger.debug("Converting ppm data to Wt%")
    df.loc[:, ppm_columns] *= scale("ppm", "Wt%")
    df.columns = [(i[0], "wt%") if i in ppm_columns else i for i in df.columns]
    return df


def format_GEOROC_references(content):
    """
    Parameters
    ---------
    content : :class:`str`
        Decoded string from GEOROC response.

    Returns
    -------
    :class:`pandas.DataFrame`
    """
    reflines = split_records(content)
    reflines = [line.replace('"', "") for line in reflines]
    reflines = [line.replace("\r\n", "") for line in reflines]
    reflines = [parse_citations(i) for i in reflines if i]
    return pd.DataFrame.from_records(reflines).set_index("key", drop=True)


def georoc_munge(df):
    """
    Collection of munging and feature adding functions for GEROROC data.

    Todo
    ------
        * Combine GEOL and AGE columns for geological ages
    """
    df = aggregate_element(df, to="Ti")
    df.loc[:, "GeolAge"] = df.loc[:, "Geol"].replace("None", "") + df.Age

    df.loc[:, "Lat"] = (df.LatitudeMax + df.LatitudeMin) / 2.0
    df.loc[:, "Long"] = (df.LongitudeMax + df.LongitudeMin) / 2.0
    return df
