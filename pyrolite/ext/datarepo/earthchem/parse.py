"""
Parsing for EarthChem-derived products.

1. Parse true duplicated columns (e.g. Mn column appears twice for some reason)
2. Parse duplicated chemical columns

Todo
-----
    * Ducplicate columns
    * H2O columns parsed as isotopes?
    * H2O, H2OP, H2OM
    * Both isotope ratios and individual isotopes
    * Isotope ratios with initial values (Ini)
    * Isotope ratios as deltas, epsilon values
    * Radioactive isotopes with activities
    * Indium?
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from ....util.text import titlecase
from ....geochem.parse import tochem, repr_isotope_ratio

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def _parse_earthchem_nounits():
    pass


def _parse_earthchem_withunits():
    pass


import xlrd
import csv
import os


def csv_from_excel(excelfile, delimiter=",", linesep=os.linesep):
    """
    Transforms an excel file to .csv for greater portability.
    """
    excelfile = Path(excelfile)
    wb = xlrd.open_workbook(str(excelfile))
    for n in wb.sheet_names():
        ws = wb.sheet_by_name(n)
        with open(
            excelfile.parent / "{}.csv".format(excelfile.stem + n), "w"
        ) as csvfile:
            wr = csv.writer(
                csvfile,
                delimiter=delimiter,
                quoting=csv.QUOTE_ALL,
                lineterminator=linesep,
            )
            for rownum in range(ws.nrows):
                wr.writerow(ws.row_values(rownum))


csv_from_excel()

# %% --
# the pandas defaul timport

import json

xl = Path("C:/GitHub/agu2018/agu2018/data/_datasets/8class/BAB/PetDB_BAB_nounits.xlsx")
df = pd.read_excel(xl)
# df = pd.read_csv((xl.parent / (xl.stem + "Data")).with_suffix(".csv"))
df.columns
# first - remove exact duplicate columns


def get_duplicated_columns(df):
    """
    On importing a dataset with duplicate columns, pandas will append a numerical
    index to subsequent duplicates (e.g. TI, TI.1, TI.2... etc). This function
    finds those which have been parsed as ducpliated column names and returns
    a set of likely duplicates indexed by the non-annotated name.
    """

    duplicated = {}
    parsed = []
    for i in df.columns:
        if i not in parsed:
            candidates = [
                c for c in df.columns if i in c and c.replace(i, "").startswith(".")
            ]
            if candidates:
                dup = [i] + candidates
                duplicated[i] = dup
                parsed += dup

    return duplicated


from pyrolite.util.math import isclose

# will need to do some data type conversions here
# check
for n, s in get_duplicated_columns(df).items():
    values = df.loc[:, s].apply(pd.to_numeric, errors='coerce').values
    all_equal = True
    for i in range(1, values.shape[1]):
        all_equal = all_equal & all(
            (values[:, i] == values[:, 0])
            + (np.isnan(values[:, i]) & np.isnan(values[:, 0]))
        )

    if all_equal:
        # drop the redundant columns
        df = df.drop(columns=s[1:])
    else:
        # we need to do some things to get the relevant data
        pass
        print('work needed')
df.columns

# %% --

with open("translate.json", "w") as f:
    f.write(json.dumps({i: tochem(titlecase(i)) for i in df.columns}))
#%% --
df.columns = [titlecase(c) for c in df.columns]
df.columns.tolist()
df.columns = tochem(df.columns)

df.columns
majors = [c for c in df.columns if c in select_oxides]
traces = [c for c in df.columns if c in select_elements]
isotopes = [c for c in df.columns if is_isotoperatio(c) and c in select_isotopes]
others = [c for c in df.columns if c.upper() in metadata_columns]
omitted = [c for c in df.columns if not c in (others + majors + traces + isotopes)]
logger.info("Omitting: {}".format(", ".join(omitted)))
df = df.loc[:, others + majors + traces + isotopes]
numeric_cols = majors + traces + isotopes
df.loc[:, numeric_cols] = df.loc[:, numeric_cols].apply(
    pd.to_numeric, errors="coerce", axis=1
)
# make sure compositional data is above zero
chem_ixs = [ix for ix, i in enumerate(df.columns) if i in (majors + traces)]
df.iloc[:, chem_ixs] = df.iloc[:, chem_ixs].mask(
    df.iloc[:, chem_ixs] <= 0.0, other=np.nan
)
df.columns = [titlecase(c) for c in df.columns]
df.columns = tochem(df.columns)
#return df
