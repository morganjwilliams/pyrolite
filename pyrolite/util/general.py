import contextlib
import datetime
import operator
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from .log import Handle

logger = Handle(__name__)

_FLAG_FIRST = object()


class Timewith:
    def __init__(self, name: str = ""):
        """Timewith context manager."""
        self.name = name
        self.start = time.time()
        self.checkpoints = []

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def checkpoint(self, name: str = ""):
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
        """Object returned on entry."""
        return self

    def __exit__(self, exittype, value, traceback):
        """Code to execute on exit."""
        self.checkpoint("Finished")
        self.checkpoints.append(("Finished", self.elapsed))


def temp_path(suffix: str = "") -> Path:
    """Return the path of a temporary directory."""
    directory = mkdtemp(suffix=suffix)
    return Path(directory)


@contextlib.contextmanager
def tempdir(**kwargs):
    tmp_dir = Path(mkdtemp(**kwargs))
    try:
        yield tmp_dir
    finally:
        remove_tempdir(tmp_dir)


def flatten_dict(d: dict, climb: bool = False, safemode: bool = False) -> dict:
    """
    Flattens a nested dictionary containing only string keys.

    This will work for dictionaries which don't have two equivalent
    keys at the same level. If you're worried about this, use safemode=True.

    Partially taken from https://stackoverflow.com/a/6043835.

    Parameters
    ----------
    climb: bool, `False`
        Whether to keep trunk or leaf-values, for items with the same key.
    safemode: bool, `True`
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

    sort = (
        x[:2]
        for x in sorted(
            [(pick_key(k), v, len(k)) for k, v in results], key=lambda x: x[-1]
        )
    )  # sorted by depth

    if not climb:
        # We go down the tree, and prioritise the trunk values
        items = sort
    else:
        # We prioritise the leaf values
        items = [i for i in sort][::-1]
    return dict(items)


def swap_item(startlist: list, pull: Any, push: Any) -> list:
    """
    Swap a specified item in a list for another.

    Parameters
    ----------
    startlist : list
        List to replace item within.
    pull
        Item to replace in the list.
    push
        Item to add into the list.

    Returns
    -------
    list
    """
    return [[i, push][i == pull] for i in startlist]


def copy_file(
    src: str | Path,
    dst: str | Path,
    ext: str | None = None,
    permissions: str | None = None,
):
    """
    Copy a file from one place to another.
    Uses the full filepath including name.

    Parameters
    ----------
    src : str | pathlib.Path
        Source filepath.
    dst : str | pathlib.Path
        Destination filepath or directory.
    ext : str, `None`
        Optional file extension specification.
    """
    src = Path(src)
    dst = Path(dst)

    if dst.is_dir():
        dst = dst / src.name

    if ext is not None:
        src = src.with_suffix(ext)
        dst = dst.with_suffix(ext)

    logger.debug(f"Copying from {src} to {dst}")
    with open(str(src), "rb") as fin, open(str(dst), "wb") as fout:
        shutil.copyfileobj(fin, fout)

    if permissions is not None:
        os.chmod(str(dst), permissions)


def remove_tempdir(directory: Path):
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
