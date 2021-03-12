import unittest
import sys
import logging
from io import StringIO
from pyrolite.util.log import (
    stream_log,
    ToLogger,
    Handle
)
import pyrolite.plot  # for logging checks


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


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
        logger.warning("Test Warning")

    def test_capture(self):
        # set up the stream
        with Capturing() as output:
            pass
            # trigger something with a logging output

        # check that the relevant logging output has been redirected to stdout


class TestHandle(unittest.TestCase):
    def test_default(self):
        # get the root pyrolite logger
        logger = Handle("pyrolite")
        self.assertIsInstance(logger, logging.Logger)

    def test_set_level(self):

        for level, val in zip(['DEBUG', 'INFO', 'WARNING', 'ERROR'], [10, 20, 30, 40]):
            with self.subTest(level=level, val=val):
                logger = Handle("pyrolite", level=level)
                self.assertTrue(logger.level == val)


if __name__ == "__main__":
    unittest.main()
