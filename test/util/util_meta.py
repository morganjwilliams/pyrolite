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

        for subf in [
            "Aitchison",
            "geochem",
            "models",
            "shannon",
            "timescale",
        ]:
            with self.subTest(subf=subf):
                folder = pyrolite_datafolder(subfolder=subf)
                self.assertIsInstance(folder, Path)
                self.assertTrue(folder.exists())
                self.assertTrue(folder.is_dir())


class TestStreamLog(unittest.TestCase):
    def setUp(self):
        pass

    def test_default(self):
        # set up the stream
        stream_log("pyrolite.plot")
        # check that the logger has a handler

        with Capturing() as output:
            pass
            # trigger something with a logging output

        # check that the relevant logging output has been redirected to stdout
        print(output)


# ToLoggeri

# take_me_to_the_docs

# sphinx_doi_link

# subkwargs

# inargs

# numpydoc_str_param_list

# get_additional_params

# update_docstring_references
