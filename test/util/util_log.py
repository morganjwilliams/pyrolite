import logging
import sys
import unittest
from io import StringIO

import pyrolite.plot  # for logging checks  # noqa: F401
from pyrolite.util.log import Handle, ToLogger


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class TestHandle(unittest.TestCase):
    def test_default(self):
        # get the root pyrolite logger
        for src in ["pyrolite", logging.getLogger(__name__)]:
            logger = Handle(src)
            self.assertIsInstance(logger, logging.Logger)

    def test_set_level(self):
        for level, val in zip(["DEBUG", "INFO", "WARNING", "ERROR"], [10, 20, 30, 40]):
            with self.subTest(level=level, val=val):
                logger = Handle("pyrolite", level=level)
                self.assertTrue(logger.level == val)


class TestToLogger(unittest.TestCase):
    def test_default(self):
        logger = Handle(__name__, level="DEBUG")

        with ToLogger(logger, "INFO") as f:
            f.write("Logging output from stream.")
            f.flush()


if __name__ == "__main__":
    unittest.main()
