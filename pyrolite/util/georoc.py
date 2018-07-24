import pandas as pd
import numpy as np
import requests
import re
import logging

from pyrolite.util.text import titlecase

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

def split_records(data, delimiter='\r\n'):
    """
    Splits records in a csv where quotation marks are used.
    Splits on a delimiter followed by an even number of quotation marks.
    """
    # https://stackoverflow.com/a/2787979
    return re.split(delimiter + '''(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', data)


def download_GEOROC_compilation(url):
    with requests.Session() as s:
        response = s.get(url)
        if response.status_code == requests.codes.ok:
            decoded_content = response.content.decode('latin-1')#.replace('"', "")
            data, ref = re.split("\s?References:\s+", decoded_content)

            datalines = [re.split(r'"\s?,\s?"', line) for line in data.splitlines()]
            df = pd.DataFrame(datalines[1:])
            df = df.applymap(lambda x: str(x).replace('"',  ""))
            df[0] = df[0].apply(lambda x: re.findall(r"[\d]+", x))
            cols = list(df.columns)
            cols[:len(datalines[0])] = [i.replace('"', "").replace(",","") for i in datalines[0]]
            df.columns = [titlecase(h, abbrv=['ID']) for h in cols]
            df = df.drop(index=df.index[~df.Citations.apply(lambda x: len(x))])
            df = df.dropna(how='all')
            df = df.set_index('UniqueID', drop=True)

            reflines = split_records(ref)
            reflines = [line.replace('"', "").replace('\r\n', "") for line in reflines] # remove quotation marks
            reflines = [i for i in reflines if i] # remove empty records
            reflines = [re.split(r'\s', line, maxsplit=1) for line in reflines] # split on first spacing
            refdf = pd.DataFrame(reflines)
            refdf.iloc[:, 0] = refdf.iloc[:, 0].apply(lambda x: re.findall(r"[\d]+", x)[0]).astype(int)
            refdf = refdf.set_index(0, drop=True)
            return df, refdf
        else:
            print('Failed download - bad status code at {}'.format(url))
            response.raise_for_status()
