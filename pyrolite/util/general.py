import os, sys
import re
import time
import subprocess, shutil
from tempfile import mkdtemp
import operator
import inspect
import zipfile
import timeit
from collections import Mapping
from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import logging


try:
    import httplib
except:
    import http.client as httplib


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

_FLAG_FIRST = object()


class Timewith:
    def __init__(self, name=""):
        self.name = name
        self.start = time.time()
        self.checkpoints = []

    @property
    def elapsed(self):
        return time.time() - self.start

    def checkpoint(self, name=""):
        elapsed = self.elapsed
        msg = "{time} {timer}: {checkpoint} in {elapsed:.3f} s.".format(
            timer=self.name,
            time=datetime.datetime.now().strftime("%H:%M:%S"),
            checkpoint=name,
            elapsed=elapsed,
        ).strip()
        logger.info(msg)
        self.checkpoints.append((name, elapsed))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint("Finished")
        self.checkpoints.append(("Finished", self.elapsed))


def stream_log(package_name, level="INFO"):
    """
    Stream the log from a specific package or subpackage.

    Parameters
    ----------
    package_name : str
        Name of the package to monitor logging from.
    level : str, 'INFO'
        Logging level at which to set the handler output.

    Returns
    -------
    logging.logger
        Logger for the specified package with stream handler added.
    """
    logger = logging.getLogger(package_name)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.setLevel(getattr(logging, level))
    return logger


def pyrolite_datafolder(subfolder=None):
    """Returns the path of the pyrolite data folder."""
    pth = Path(sys.modules["pyrolite"].__file__).parent / "data"
    if subfolder:
        pth /= subfolder
    return pth


def pathify(path):
    """Converts strings to pathlib.Path objects."""
    if not isinstance(path, Path):
        path = Path(path)
    return path


def urlify(url):
    """Strip a string to return a valid URL."""
    return url.strip().replace(" ", "_")


def internet_connection(target="www.google.com"):
    """
    Tests for an active internet connection, based on an optionally specified
    target.

    Parameters
    ----------
    target: url
        URL to check connectivity, defaults to www.google.com

    Returns
    -------
    bool
        Boolean indication of whether a HTTP connection can be established at the given
        url.
    """
    conn = httplib.HTTPConnection(target, timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False


def temp_path(suffix=""):
    """Return the path of a temporary directory."""
    dir = mkdtemp(suffix=suffix)
    return Path(dir)


def iscollection(obj):
    """
    Checks whether an object is an iterable collection.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        Boolean indication of whether the object is a collection.
    """

    for ty in [list, np.ndarray, set, tuple, dict, pd.Series]:
        if isinstance(obj, ty):
            return True

    return False


def check_perl():
    """
    Checks whether perl is installed on the system.

    Returns
    -------
    bool
        Boolean indication of whether there is an executable perl installation.
    """
    try:
        p = subprocess.check_output("perl -v")
        returncode = 0
    except subprocess.CalledProcessError as e:
        output = e.output
        returncode = e.returncode
    except FileNotFoundError:
        returncode = 1.0

    return returncode == 0


def flatten_dict(d, climb=False, safemode=False):
    """
    Flattens a nested dictionary containing only string keys.

    This will work for dictionaries which don't have two equivalent
    keys at the same level. If you're worried about this, use safemode=True.

    Partially taken from https://stackoverflow.com/a/6043835.

    Parameters
    ----------
    climb: True | False, False
        Whether to keep trunk or leaf-values, for items with the same key.
    safemode: True | False, False
        Whether to keep all keys as a tuple index, to avoid issues with
        conflicts.

    Returns
    -------
    dict
        Flattened dictionary.
    """
    lift = lambda x: (x,)
    join = operator.add
    results = []

    def visit(subdict, results, partialKey):
        for k, v in subdict.items():
            if partialKey == _FLAG_FIRST:
                newKey = lift(k)
            else:
                newKey = join(partialKey, lift(k))
            if isinstance(v, Mapping):
                visit(v, results, newKey)
            else:
                results.append((newKey, v))

    visit(d, results, _FLAG_FIRST)

    if safemode:
        pick_key = lambda keys: keys
    else:
        pick_key = lambda keys: keys[-1]

    sort = map(
        lambda x: x[:2],
        sorted([(pick_key(k), v, len(k)) for k, v in results], key=lambda x: x[-1]),
    )  # sorted by depth

    if not climb:
        # We go down the tree, and prioritise the trunk values
        items = sort
    else:
        # We prioritise the leaf values
        items = [i for i in sort][::-1]
    return dict(items)


def swap_item(list: list, pull: object, push: object):
    """
    Swap a specified item in a list for another.

    Parameters
    ----------
    list : list
        List to replace item within.
    pull : object
        Item to replace in the list.
    push : object
        Item to add into the list.

    Returns
    -------
    list
    """
    return [[i, push][i == pull] for i in list]


def copy_file(src, dst, ext=None):
    """
    Copy a file from one place to another.
    Uses the full filepath including name.

    Parameters
    ----------
    src : str | Path
        Source filepath.
    dst : str | Path
        Destination filepath.
    ext : extension, None
        Optional file extension specification.
    """
    src = Path(src)
    dst = Path(dst)
    if ext is not None:
        src = src.with_suffix(ext)
        dst = dst.with_suffix(ext)
    logger.debug("Copying from {} to {}".format(src, dst))
    with open(str(src), "rb") as fin:
        with open(str(dst), "wb") as fout:
            shutil.copyfileobj(fin, fout)


def remove_tempdir(directory):
    """
    Remove a specific directory, contained files and sub-directories.

    Parameters
    ----------
    directory: str, Path
        Path to directory.
    """
    directory = Path(directory)
    shutil.rmtree(str(directory))
    assert not directory.exists()


def extract_zip(zipfile, output_dir):
    """
    Extracts a zipfile without the uppermost folder.

    Parameters
    ----------
    zipfile: zipfile object
        Zipfile object to extract.
    output_dir: str | Path
        Directory to extract files to.
    """
    output_dir = Path(output_dir)
    if zipfile.testzip() is None:
        for m in zipfile.namelist():
            fldr, name = re.split("/", m, maxsplit=1)
            if name:
                content = zipfile.open(m, "r").read()
                with open(str(output_dir / name), "wb") as out:
                    out.write(content)
