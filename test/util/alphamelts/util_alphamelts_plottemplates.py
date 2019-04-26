import unittest
import pandas as pd
import io
from pyrolite.util.pd import to_numeric
from pyrolite.util.synthetic import test_df, test_ser
from pyrolite.util.alphamelts.download import install_melts
from pyrolite.util.general import check_perl, temp_path, remove_tempdir
from pyrolite.util.alphamelts.automation import *
from pyrolite.util.alphamelts.plottemplates import *

_env = (
    pyrolite_datafolder(subfolder="alphamelts")
    / "localinstall"
    / "examples"
    / "alphamelts_default_env.txt"
)

_melts = (
    pyrolite_datafolder(subfolder="alphamelts")
    / "localinstall"
    / "examples"
    / "Morb.melts"
)

if __name__ == "__main__":
    if not (pyrolite_datafolder(subfolder="alphamelts") / "localinstall").exists():
        install_melts(local=True)  # install melts for example files etc

    unittest.main()
