import os, sys
import re
import time
import subprocess, shutil
import operator
import inspect
import zipfile
import timeit
from collections import Mapping
from pathlib import Path
import numpy as np
import pandas as pd
import logging

try:
    import httplib
except:
    import http.client as httplib
import pyrolite  # required for introspection/data folder


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

_FLAG_FIRST = object()


class Timewith:
    def __init__(self, name=""):
        self.name = name
        self.start = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    def checkpoint(self, name=""):
        msg = "{timer:.3f} {checkpoint:.3f} in {elapsed:.3f} s.".format(
            timer=self.name, checkpoint=name, elapsed=self.elapsed
        ).strip()
        print(msg)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint("Finished")
        pass


def stream_log(package_name, level="INFO"):
    """
    Stream the log from a specific package or subpackage.
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
    pth = Path(inspect.getfile(pyrolite)).parent / "data"
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
    """
    conn = httplib.HTTPConnection(target, timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False


def temp_path():
    """Return the path of a temporary directory."""
    userdir = Path("~").expanduser()
    root = Path(userdir.drive) / userdir.root
    if root / "tmp" in root.iterdir():  # .nix
        return root / "tmp"
    else:
        return root / "temp"


def iscollection(obj):
    """Checks whether an object is an interable collection"""

    for ty in [list, np.ndarray, set, tuple, dict, pd.Series]:
        if isinstance(obj, ty):
            return True

    return False


def check_perl():
    """Checks whether perl is installed on the system."""
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


def swap_item(list: list, pull: str, push: str):
    """
    Swap a specified item in a list for another.

    Parameters
    ----------
    list: List to replace item within.
    pull: Item to replace in the list.
    push: Item to add into the list.
    """
    return [push if i == pull else i for i in list]


def copy_file(src, dst, ext=None):
    """
    Copy a file from one place to another.
    Uses the full filepath including name.

    Parameters
    ----------
    src: str | Path
        Source filepath.
    dst: str | Path
        Destination filepath.
    ext: extension, None
        Optional file extension specification.
    """
    src = Path(src)
    dst = Path(dst)
    if ext is not None:
        src = src.with_suffix(ext)
        dst = dst.with_suffix(ext)
    print("Copying from {} to {}".format(src, dst))
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
    if directory.exists():
        temp_files = []
        for x in directory.iterdir():
            if x.is_file():
                temp_files.append(x)
            elif x.is_dir():
                remove_tempdir(x)
        for t in temp_files:
            os.remove(str(t))
        os.rmdir(str(directory))
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
