import sys
import unittest
from pathlib import Path
from io import StringIO
from pyrolite.util.meta import (
    pyrolite_datafolder,
    stream_log,
    ToLogger,
    take_me_to_the_docs,
    sphinx_doi_link,
    subkwargs,
    inargs,
    numpydoc_str_param_list,
    get_additional_params,
    update_docstring_references,
)
import pyrolite.plot  # for logging checks
from pyrolite.util.synthetic import test_df
import logging


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class TestPyroliteDataFolder(unittest.TestCase):
    def test_default(self):
        folder = pyrolite_datafolder()
        self.assertIsInstance(folder, Path)
        self.assertTrue(folder.exists())
        self.assertTrue(folder.is_dir())

    def test_subfolders(self):

        for subf in ["Aitchison", "geochem", "models", "shannon", "timescale"]:
            with self.subTest(subf=subf):
                folder = pyrolite_datafolder(subfolder=subf)
                self.assertIsInstance(folder, Path)
                self.assertTrue(folder.exists())
                self.assertTrue(folder.is_dir())


class TestStreamLog(unittest.TestCase):
    def setUp(self):
        self.key = "pyrolite.plot"

    def test_default(self):
        logger = stream_log(self.key)
        # check that the logger has a single stream handler
        handlers = [i for i in logger.handlers if isinstance(i, logging.StreamHandler)]
        self.assertTrue(len(handlers) == 1)

    def test_INFO(self):
        level = "INFO"
        logger = stream_log(self.key, level=level)
        # check that the logger has a single stream handler
        handlers = [i for i in logger.handlers if isinstance(i, logging.StreamHandler)]
        self.assertTrue(len(handlers) == 1)
        self.assertTrue(handlers[0].level == getattr(logging, level))

    def test_WARNING(self):
        level = "WARNING"
        logger = stream_log(self.key, level=level)
        # check that the logger has a single stream handler
        handlers = [i for i in logger.handlers if isinstance(i, logging.StreamHandler)]
        self.assertTrue(len(handlers) == 1)
        self.assertTrue(handlers[0].level == getattr(logging, level))

    def test_ouptut(self):
        level = "WARNING"
        logger = stream_log(self.key, level=level)
        logger.warning('Test Warning')

    def test_capture(self):
        # set up the stream
        with Capturing() as output:
            pass
            # trigger something with a logging output

        # check that the relevant logging output has been redirected to stdout


# ToLogger

# take_me_to_the_docs

# sphinx_doi_link

# subkwargs

# inargs

# numpydoc_str_param_list

# get_additional_params

# update_docstring_references
