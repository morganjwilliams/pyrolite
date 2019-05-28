import logging
import json
import requests
import urllib
from pathlib import Path
from bs4 import BeautifulSoup
from http.client import HTTPResponse

from ....util.text import titlecase
from ....util.general import temp_path
from ....util.meta import pyrolite_datafolder
from ....util.web import urlify, internet_connection, download_file

from .schema import parse_GEOROC_response

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__contents_file__ = pyrolite_datafolder(subfolder="georoc") / "contents.json"


def bulk_download(
    output_folder=Path("~/Downloads/GEOROC"), collections=None, redownload: bool = False
):
    """
    Download utility for GEOROC data. Facilitates incremental and resumed
    downloads. Output data will be organised into folders by reservoir, and
    stored as individual CSVs.

    Parameters
    ----------
    output_folder : :class:`str` | :class:`pathlib.Path`
        Path to folder to store output data.
    collections : :class:`list`, :code:`None`
        List of names (e.g. 'ConvergentMargins') or abbreviations (e.g. 'CM') for
        GEOROC compilations to download.
    redownload : :class:`bool`, :code:`False`
        Whether to redownload prevoiusly downloaded compilations.
    """

    output_folder = output_folder or temp_path()
    output_folder = Path(output_folder)
    output_folder = output_folder.expanduser()

    update_filelist()

    # construct a dictionary of form {name : filelist}
    if collections is None:  # unspecified, get everything
        __colls__ = __CONTENTS__
    elif isinstance(collections, dict):
        # get all the dictionary values are dictionaries, try to get the files attr
        __colls__ = {c: getattr(v, "files", v) for c, v in collections.items()}
    elif isinstance(collections, (list, tuple)):
        if isinstance(collections[0], (list, tuple)):  # list of lists
            # specifiying collections and list of files
            __colls__ = {c: v for c, v in collections}
        else:  # list of names
            # subset of georoc collections
            __colls__ = {}
            abbrvs = {__CONTENTS__[c]["abbrv"]: __CONTENTS__[c] for c in __CONTENTS__}
            subset = [i for i in collections if i in __CONTENTS__ or i in abbrvs]
            for c in collections:
                if c in __CONTENTS__:
                    __colls__ = {**__colls__, **{c: __CONTENTS__[c]["files"]}}
                elif c in abbrvs:
                    __colls__ = {**__colls__, **{c: abbrvs[c]["files"]}}
                else:
                    logger.warn("Collection not recognised : {}".format(c))
    else:
        logger.warn("Format of collections not recognised.")

    logger.info("Beginning bulk download for {}.".format(", ".join(__colls__.keys())))

    completed = []
    for res in __colls__:
        resdir = output_folder / res
        if not resdir.exists():
            resdir.mkdir(parents=True)

        files = __colls__[res]  # Compilation List of Targets
        base_url = r"http://georoc.mpch-mainz.gwdg.de" + "/georoc/Csv_Downloads"

        if not redownload:
            # Just get the ones we don't have, continuing from last 'save'
            logger.info("Downloading only undownloaded files.")
            stems = [(resdir / urlify(f)).stem for f in files]
            current_files = [f.stem for f in resdir.iterdir() if f.is_file()]
            files = [f for f, s in zip(files, stems) if not s in current_files]

        dataseturls = [
            (urlify(d), base_url + r"/" + urlify(d)) for d in files if d.strip()
        ]

        for name, url in dataseturls:
            if "/" in name:
                name = name.split("/")[-1]
            outfile = (resdir / name).with_suffix("")
            msg = "Downloading {} {} dataset to {}.".format(res, name, outfile)
            logger.info(msg)
            try:
                df = download_file(
                    url, encoding="latin-1", postprocess=parse_GEOROC_response
                )
                df.to_csv(outfile.with_suffix(".csv"))
            except requests.exceptions.HTTPError as e:
                pass
        logger.info("Download and aggregation for {} finished.".format(res))
        completed.append(res)
    logger.info("Bulk download for {} completed.".format(", ".join(completed)))


def get_filelinks(
    page="http://georoc.mpch-mainz.gwdg.de/georoc/CompFiles.aspx",
    exclude=["Minerals", "Rocks", "Inclusions", "Georoc"],
):
    """
    Get the links for all compliation files from the GEOROC website.

    Parameters
    ------------
    page : :class:`str` | :class:`http.client.HTTPResponse`
        String URL or http.client.HTTPResponse to scrape for links.
    exclude : :class:`list`
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


def update_filelist(filepath=pyrolite_datafolder(subfolder="georoc") / "contents.json"):
    """
    Update a local copy listing the compilations available from GEOROC.

    Parameters
    ----------
    filepath : :class:`str` | :class:`pathlib.Path`
        Path at which to save the filelist dictionary.

    Returns
    -------
    :class:`dict`
        Dictionary of GEOROC section compilations.
    """
    try:
        assert internet_connection(target="georoc.mpch-mainz.gwdg.de")
        contents = get_filelinks()

        with open(str(filepath), "w+") as fh:
            fh.write(json.dumps(contents))
    except AssertionError:
        msg = "Unable to make onnection to GEOROC to update compilation lists."
        logger.warning(msg)

    return contents


if __contents_file__.exists():
    with open(str(__contents_file__)) as fh:
        __CONTENTS__ = json.loads(fh.read())
else:
    if not __contents_file__.parent.exists():
        __contents_file__.parent.mkdir(parents=True)
    __CONTENTS__ = {}
    update_filelist()
    with open(str(__contents_file__)) as fh:
        __CONTENTS__ = json.loads(fh.read())
