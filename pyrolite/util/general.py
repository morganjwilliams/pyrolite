import os, sys
import psutil
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


def pathify(path):
    """Converts strings to pathlib.Path objects."""
    if not isinstance(path, Path):
        path = Path(path)
    return path


def temp_path(suffix=""):
    """Return the path of a temporary directory."""
    dir = mkdtemp(suffix=suffix)
    return Path(dir)


def check_perl():
    """
    Checks whether perl is installed on the system.

    Returns
    -------
    :class:`bool`
        Boolean indication of whether there is an executable perl installation.
    """
    try:
        p = subprocess.check_output(["perl", "-v"])
        returncode = 0
    except subprocess.CalledProcessError as e:
        output = e.output
        returncode = e.returncode
    except FileNotFoundError:
        returncode = 1

    return returncode == 0


def flatten_dict(d, climb=False, safemode=False):
    """
    Flattens a nested dictionary containing only string keys.

    This will work for dictionaries which don't have two equivalent
    keys at the same level. If you're worried about this, use safemode=True.

    Partially taken from https://stackoverflow.com/a/6043835.

    Parameters
    ----------
    climb: :class:`bool`, :code:`False`
        Whether to keep trunk or leaf-values, for items with the same key.
    safemode: :class:`bool`, :code:`True`
        Whether to keep all keys as a tuple index, to avoid issues with
        conflicts.

    Returns
    -------
    :class:`dict`
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
    list : :class:`list`
        List to replace item within.
    pull
        Item to replace in the list.
    push
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
    src : :class:`str` | :class:`pathlib.Path`
        Source filepath.
    dst : :class:`str` | :class:`pathlib.Path`
        Destination filepath.
    ext : :class:`str`, :code:`None`
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
    try:
        shutil.rmtree(str(directory))
        assert not directory.exists()
    except PermissionError:
        pass


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


def get_process_tree(process, levels_up=1):
    """
    Get a process tree from an active process or process ID.

    Parameters
    -----------
    process : :class:`int` | :class:`psutil.Process`
        Process to search for.
    levels_up : :class:`int`
        How many levels up the tree to search for parent processes.

    Returns
    -------
    :class:`list`
        List of processes associated with the given process tree.
    """
    if isinstance(process, int):
        top = psutil.Process(process)
    elif isinstance(process, psutil.Process):
        top = process
    for i in range(levels_up):
        if top.parent() is not None:
            top = top.parent()
        else:
            break
    return [top, *top.children(recursive=True)]
