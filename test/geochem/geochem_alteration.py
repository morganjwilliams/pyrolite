import unittest

import numpy as np
import pandas as pd

from pyrolite.geochem.alteration import (
    CCPI,
    CIA,
    CIW,
    PIA,
    SAR,
    WIP,
    IshikawaAltIndex,
    SiTiIndex,
)


class TestAlterationIndicies(unittest.TestCase):
    """Tests the chemical index of alteration measure."""

    def setUp(self):
        self.cols = ["SiO2", "CaO", "MgO", "FeO", "TiO2", "Na2O", "K2O", "Al2O3"]
        self.df = pd.DataFrame(
            {k: v for k, v in zip(self.cols, np.random.rand(len(self.cols), 10))}
        )

    def test_CIA_default(self):
        """Tests the chemical index of alteration measure."""
        df = self.df
        df["CIA"] = CIA(df)

    def test_CIW_default(self):
        """Tests the chemical index of weathering measure."""
        df = self.df
        df["CIW"] = CIW(df)

    def test_PIA_default(self):
        """Tests the plagioclase index of alteration measure."""
        df = self.df
        df["PIA"] = PIA(df)

    def test_SAR_default(self):
        """Tests the silica alumina ratio measure."""
        df = self.df
        df["SAR"] = SAR(df)

    def test_SiTiIndex_default(self):
        """Tests the silica titania ratio measure."""
        df = self.df
        df["SiTiIndex"] = SiTiIndex(df)

    def test_WIP_default(self):
        """Tests the weathering index of parker measure."""
        df = self.df
        df["WIP"] = WIP(df)

    def test_AI_default(self):
        """Tests the Alteration index of Ishikawa."""
        df = self.df
        df["IshikawaAltIndex"] = IshikawaAltIndex(df)

    def test_CCPI_default(self):
        """Tests the Chlorite-carbonate-pyrite index."""
        df = self.df
        df["CCPI"] = CCPI(df)


if __name__ == "__main__":
    unittest.main()
