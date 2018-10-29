import pandas as pd
import numpy as np
import requests
import re
import logging

from pyrolite.util.pd import *
from pyrolite.util.text import titlecase, parse_entry, split_records
from pyrolite.util.general import temp_path, urlify, pyrolite_datafolder, \
                                  pathify
from pyrolite.geochem import tochem, check_multiple_cation_inclusion, \
                             aggregate_cation

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# -----------------------------
# GEOROC INFO
# -----------------------------

_GEOROC_value_rx = r"(\s)*?(?P<value>[\.\w]+)((\s)*?\[)?(?P<key>\w*)(\])?(\s)*?"
_GEOROC_cit_rx = r"(\s)*?(\[)?(?P<key>\w*)(\])?(\s)*?(?P<value>[\.\w]+)(\s)*?"
_GEOROC_full_cit_rx = r"(\s)*?(\[)?(?P<key>\w*)(\])?(\s)*?(?P<value>.*)(\s)*?"

__GEOROC_contents = {
'CFB': dict(url_suffix=r'Continental_Flood_Basalts_comp',
            list_file="GEOROC_CFB_Dataset_List.csv"),
'ConvergentMargins': dict(url_suffix=r'Convergent_Margins_comp',
                          list_file="GEOROC_Convergent_Dataset_List.csv"),
'OceanicPlateaus': dict(url_suffix=r'Oceanic_Plateaus_comp',
                        list_file="GEOROC_OceanicPlateau_Dataset_List.csv"),
'OIB': dict(url_suffix=r'Ocean_Island_Groups_comp',
            list_file="GEOROC_OIB_Dataset_List.csv"),
        }


__compilation_lists = pyrolite_datafolder(subfolder='georoc')
# -----------------------------


def bulk_GEOROC_download(output_folder=Path('~/Downloads/GEOROC'),
                         reservoirs=None,
                         redownload=True):

    output_folder = output_folder or temp_path()
    output_folder = pathify(output_folder)
    output_folder = output_folder.expanduser()

    reservoirs = reservoirs or __GEOROC_contents.keys()

    for k in reservoirs:
        resdir = output_folder / k
        v = __GEOROC_contents[k]

        if not resdir.exists():
            resdir.mkdir(parents=True)

        # Compilation List of Targets
        list_file = __compilation_lists / v['list_file']
        filenames = [i.split(',')[0].strip() for i in open(list_file).readlines()]

        # URL target
        host = r'http://georoc.mpch-mainz.gwdg.de/georoc/Csv_Downloads'
        base_url = host + "/" + v['url_suffix']

        # Files yet to download, continuing from last 'save'
        download_filenames = filenames
        if not redownload:
            logger.info('Fetching only undownloaded files.')
            # Just get the ones we don't have
            download_filenames = [f for f in download_filenames
                                  if not (resdir / urlify(f)).exists()]
        dataseturls = [(urlify(d), base_url + r"/" + urlify(d))
                       for d in download_filenames if d if d.strip()]

        for name, url in dataseturls:
            logger.info('Downloading {} {} dataset.'.format(k, name))
            try:
                df, ref = download_GEOROC_compilation(url)
                df.to_csv(resdir / name)
            except requests.exceptions.HTTPError:
                pass

        # Compile CSVs
        DF = df_from_csvs(resdir.glob('*.csv'), ignore_index=True)
        size = DF.index.size
        logger.info('Aggregated {} datasets ({} records).'.format(k, size))

        # Save the compilation
        out_file = output_folder / ('GEOROC_' + k)
        sparse_pickle_df(DF, out_file)

    logger.info('Download and aggregation for {} finished.'.format(k))


def download_GEOROC_compilation(url):
    with requests.Session() as s:
        response = s.get(url)
        if response.status_code == requests.codes.ok:
            logger.info('Response recieved from {}.'.format(url))
            decoded_content = response.content.decode('latin-1')
            data, ref = re.split("\s?References:\s+", decoded_content)

            datalines = [re.split(r'"\s?,\s?"', line)
                         for line in data.splitlines()]
            df = pd.DataFrame(datalines[1:])
            df = df.applymap(lambda x: str(x).replace('"',  ""))
            df[0] = df[0].apply(lambda x: re.findall(r"[\d]+", x))
            cols = list(df.columns)
            cols[:len(datalines[0])] = [i.replace('"', "").replace(",","")
                                        for i in datalines[0]]
            df.columns = [titlecase(h, abbrv=['ID']) for h in cols]
            df = df.drop(index=df.index[~df.Citations.apply(lambda x: len(x))])
            df = df.dropna(how='all')
            df = df.set_index('UniqueID', drop=True)

            reflines = split_records(ref)
            # remove quotation marks and newlines, remove empty records
            reflines = [line.replace('"', "") for line in reflines]
            reflines = [line.replace('\r\n', "") for line in reflines]
            reflines = [i for i in reflines if i]
            # split on first spacing
            reflines = [re.split(r'\s', line, maxsplit=1) for line in reflines]
            refdf = pd.DataFrame(reflines)
            refdf.iloc[:, 0] = refdf.iloc[:, 0].apply(lambda x:
                                                      re.findall(r"[\d]+", x)[0]
                                                      ).astype(int)
            refdf = refdf.set_index(0, drop=True)
            return df, refdf
        else:
            logger.warning('Failed download - bad status code at {}'.format(url))
            response.raise_for_status()


def ingest_pickled_georoc_frame(path):
    df = load_sparse_pickle_df(path)

    start = list(df.columns).index('Material')
    end = list(df.columns).index('Nd143Nd144')
    headercols = df.columns[:start+1]
    chemcols = df.columns[start+1:end-1]
    trailingcols = df.columns[end-1:]
    # units conversion
    df = to_numeric(df, exclude=list(headercols))
    # trailing are generally isotope ratios
    where_ppm = [(('ppm' in i) and (i in chemcols)) for i in df.columns]
    df.loc[:, where_ppm] = df.loc[:, where_ppm] / 10000 # in wt%
    df.columns = tochem([c.replace("(wt%)", "").replace("(ppm)", "")
                         for c in df.columns])
    chemcols = df.columns[start+1:end-1]

    duplicate_chemcols = df.columns[df.columns.duplicated() & \
                                    [i in chemcols for i in df.columns]]

    for c in duplicate_chemcols:
        subdf = df[c]
        ser = subdf.apply(np.nansum, axis=1)
        df = df.drop(columns=[c])
        df[c] = ser

    # the length of the column index has changed...
    chemcols = chemcols.unique()

    for c in chemcols:
        df[c] = df.loc[:, c].where(df.loc[:, c] > 0., other=np.nan) # remove <0.

    mulitiple_cations = check_multiple_cation_inclusion(df)
    df = aggregate_cation(df, 'Ti', form='element')

    chemcols = [i for i in chemcols if i in df.columns]
    df[chemcols] = df[chemcols] * 10000. # everything in ppm

    # text fields to parse
    _parse = ['Age']

    return df
