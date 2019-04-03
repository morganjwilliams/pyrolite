import logging
import json
import requests
import urllib
from pathlib import Path
from bs4 import BeautifulSoup
from http.client import HTTPResponse

from ...text import titlecase
from ...general import temp_path, pyrolite_datafolder
from ...web import urlify, internet_connection, download_file

from .schema import format_GEOROC_response

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__contents_file__ = pyrolite_datafolder(subfolder="georoc") / "contents.json"


def bulk_GEOROC_download(
    output_folder=Path("~/Downloads/GEOROC"),
    reservoirs=None,
    redownload: bool = False,
    write_hdf: bool = False,
    write_pickle: bool = False,
):
    """
    Download utility for GEOROC data. Facilitates incremental and resumed
    downloadsself. Output data will be organised into folders by reservoir, and
    stored as both i) individual CSVs and ii) a picked pd.DataFrame.

    Notes
    -----
        Chemical abundance data are output as Wt% by default.


    Parameters
    ----------
    output_folder : :class:`str` | :class:`pathlib.Path`
        Path to folder to store output data.
    reservoirs : :class:`list`, :code:`None`
        List of names (e.g. 'ConvergentMargins') or abbrevaitions (e.g. 'CM') for
        GEOROC compilations to download.
    redownload : :class:`bool`, :code:`False`
        Whether to redownload prevoiusly downloaded compilations.
    write_hdf : :class:`bool`, :code:`False`
        Whether to create HDF5 files for each compilation.
    write_pickle : :class:`bool`, :code:`False`
        Whether to create pickle files for each compilation.
    """

    output_folder = output_folder or temp_path()
    output_folder = Path(output_folder)
    output_folder = output_folder.expanduser()

    update_georoc_filelist()

    reservoirs = reservoirs or __CONTENTS__.keys()
    abbrvs = {__CONTENTS__[k]["abbrv"]: k for k in __CONTENTS__}

    if not redownload:
        logger.info("Bulk download for {} beginning.".format(", ".join(reservoirs)))

    completed = []
    for res in reservoirs:
        if res in __CONTENTS__.keys():
            resname = res
            resabbrv = v["abbrv"]
        elif res in abbrvs:
            resname = abbrvs[res]
            resabbrv = res
        else:
            msg = "Unknown reservoir requested: {}".format(res)
            logger.warn(msg)
        if resname:
            v = __CONTENTS__[resname]

            resdir = output_folder / res
            if not resdir.exists():
                resdir.mkdir(parents=True)

            out_aggfile = resdir / ("_" + res)

            # Compilation List of Targets
            filenames = v["files"]

            # URL target
            host = r"http://georoc.mpch-mainz.gwdg.de"
            base_url = host + "/georoc/Csv_Downloads"

            # Files yet to download, continuing from last 'save'
            dwnld_fns = filenames
            if not redownload:
                # Just get the ones we don't have,
                logger.info("Downloading only undownloaded files.")
                dwnld_stems = [(resdir / urlify(f)).stem for f in dwnld_fns]
                current_files = [f.stem for f in resdir.iterdir() if f.is_file()]
                dwnld_fns = [
                    f for f, s in zip(dwnld_fns, dwnld_stems) if not s in current_files
                ]

            dataseturls = [
                (urlify(d), base_url + r"/" + urlify(d)) for d in dwnld_fns if d.strip()
            ]

            for name, url in dataseturls:
                if "/" in name:
                    name = name.split("/")[-1]
                outfile = (resdir / name).with_suffix("")
                msg = "Downloading {} {} dataset to {}.".format(res, name, outfile)
                logger.info(msg)
                try:
                    df = download_file(
                        url, encoding="latin-1", postprocess=format_GEOROC_response
                    )
                    df.to_csv(outfile.with_suffix(".csv"))
                except requests.exceptions.HTTPError as e:
                    pass

            if write_hdf or write_pickle:
                aggdf = df_from_csvs(resdir.glob("*.csv"), ignore_index=True)
                msg = "Aggregated {} datasets ({} records).".format(
                    res, aggdf.index.size
                )
                logger.info(msg)

                # Save the compilation
                if write_pickle:
                    sparse_pickle_df(aggdf, out_aggfile)

                if write_hdf:
                    min_itemsize = {
                        c: 100 for c in aggdf.columns[aggdf.dtypes == "object"]
                    }
                    min_itemsize.update({"Citations": 1200})
                    aggdf.to_hdf(
                        out_aggfile.with_suffix(".h5"),
                        out_aggfile.stem,
                        min_itemsize=min_itemsize,
                        mode="w",
                    )

            logger.info("Download and aggregation for {} finished.".format(res))
            completed.append(res)
    logger.info("Bulk download for {} completed.".format(", ".join(completed)))


def get_georoc_links(
    page="http://georoc.mpch-mainz.gwdg.de/georoc/CompFiles.aspx",
    exclude=["Minerals", "Rocks", "Inclusions", "Georoc"],
):
    """
    Parameters
    ------------
    page: {str, HTTPResponse}
        String URL or http.client.HTTPResponse to scrape for links.
    exclude: list
        List of collections not to get links for.
    """
    if isinstance(page, str):
        page = urllib.request.urlopen(page)

    soup = BeautifulSoup(page, "html.parser")
    links = [
        link.get("href")
        for link in list(soup.find_all("a"))
        if not link.get("href") is None
    ]
    pathlinks = [Path(i) for i in links if "_comp" in i]
    groups = set([l.parent.name for l in pathlinks])
    contents = {}
    for g in groups:
        name = titlecase(g.replace("_comp", "").replace("_", " "))
        if name not in exclude:
            abbrv = "".join([s for s in g if s == s.upper() and not s in ["_", "-"]])
            # File names which include url_suffix:
            grp = ["".join([g, "/", i.name]) for i in pathlinks if i.parent.name == g]
            contents[name] = {"files": grp, "abbrv": abbrv}

    return contents


def update_georoc_filelist(
    filepath=pyrolite_datafolder(subfolder="georoc") / "contents.json"
):
    """
    Update a local copy listing the compilations available from GEOROC.
    """
    try:
        assert internet_connection(target="georoc.mpch-mainz.gwdg.de")
        contents = get_georoc_links()

        with open(str(filepath), "w+") as fh:
            fh.write(json.dumps(contents))
    except AssertionError:
        msg = "Unable to make onnection to GEOROC to update compilation lists."
        logger.warning(msg)


if __contents_file__.exists():
    with open(str(__contents_file__)) as fh:
        __CONTENTS__ = json.loads(fh.read())
else:
    if not __contents_file__.parent.exists():
        __contents_file__.parent.mkdir(parents=True)
    __CONTENTS__ = {}
    update_georoc_filelist()
    with open(str(__contents_file__)) as fh:
        __CONTENTS__ = json.loads(fh.read())
