import os, sys
from collections import Mapping
from itertools import chain
import operator
import numpy as np
import logging
import shutil
from pathlib import Path
import zipfile
import re

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger()


_FLAG_FIRST = object()


def flatten_dict(d, climb=False, safemode=False):
    """
    Flattens a nested dictionary containing only string keys.

    This will work for dictionaries which don't have two equivalent
    keys at the same level. If you're worried about this, use safemode=True.

    Partially taken from https://stackoverflow.com/a/6043835.

    Parameters
    ----------
    climb: True | False
        Whether to keep trunk or leaf-values, for items with the same key.
    """
    lift = lambda x: (x, )
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

    sort = map(lambda x: x[:2],
               sorted([(pick_key(k), v, len(k)) for k, v in results],
                      key=lambda x: x[-1]) ) # sorted by depth

    if not climb:
        # We go down the tree, and prioritise the trunk values
        items = sort
    else:
        # We prioritise the leaf values
        items = [i for i in sort][::-1]
    return dict(items)


def swap_item(list: list, pull: str, push: str):
    return [push if i == pull else i for i in list]


def copy_file(src, dst, ext=None):
    src = Path(src)
    dst = Path(dst)
    if ext is not None:
        src = src.with_suffix(ext)
        dst = dst.with_suffix(ext)
    print('Copying from {} to {}'.format(src, dst))
    with open(src, 'rb') as fin:
        with open(dst, 'wb') as fout:
            shutil.copyfileobj(fin, fout)


def remove_tempdir(directory):
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
    """Extracts a zipfile without the uppermost folder."""
    output_dir = Path(output_dir)
    if zipfile.testzip() is None:
        for m in zipfile.namelist():
            fldr, name = re.split('/', m, maxsplit=1)
            if name:
                content = zipfile.open(m, 'r').read()
                with open(output_dir / name, 'wb') as out:
                    out.write(content)
