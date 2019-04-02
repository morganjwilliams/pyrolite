import unittest
import pandas as pd
import io
from pyrolite.util.pd import to_numeric
from pyrolite.util.synthetic import test_df, test_ser
from pyrolite.util.alphamelts.meltsfile import *

def str_as_file(str):
    return io.BytesIO(str.encode('UTF-8'))

KM0417_RC12_ser = to_numeric(
    pd.DataFrame(
        [
            ("Title", "KM0417_RC12"),
            ("SiO2", 51.27),
            ("TiO2", 0.96),
            ("Al2O3", 0.96),
            ("Fe2O3", 1.1),
            ("Cr2O3", 0.042),
            ("FeO", 8.70),
            ("MnO", 0.176),
            ("MgO", 8.45),
            ("NiO", 0.011),
            ("CaO", 12.47),
            ("Na2O", 2.25),
            ("K2O", 0.047),
            ("P2O5", 0.065),
            ("H2O", 0.25),
            ("CO2", 0.005),
            ("Initial Pressure", 100),
            ("Final Pressure", 1),
            ("Increment Pressure", -1),
            ("Initial Temperature", 1190.66),
            ("Final Temperature", 1190.66),
            ("Increment Temperature", 0.0),
            ("Log fO2 Path", "FMQ"),
            ("Log fO2 Offset", -1),
            ("Mode", "Fractionate Fluids"),
        ]
    ).set_index(0, drop=True),
    errors="ignore",
)[1]

KM0417_RC12 = """Title: KM0417_RC12
Initial Composition: SiO2 51.27
Initial Composition: TiO2 0.96
Initial Composition: Al2O3 0.96
Initial Composition: Fe2O3 1.1
Initial Composition: Cr2O3 0.042
Initial Composition: FeO 8.70
Initial Composition: MnO 0.176
Initial Composition: MgO 8.45
Initial Composition: NiO 0.011
Initial Composition: CaO 12.47
Initial Composition: Na2O 2.25
Initial Composition: K2O 0.047
Initial Composition: P2O5 0.065
Initial Composition: H2O 0.25
Initial Composition: CO2 0.005
Initial Pressure: 100
Final Pressure: 1
Increment Pressure: -1
Initial Temperature: 1190.66
Final Temperature: 1190.66
Increment Temperature: 0
Log fO2 Path: FMQ
Log fO2 Offset: -1
Mode: Fractionate Fluids
"""

WH_DMM_with_REE_ser = pd.DataFrame(
    [
        ("Title", "Depleted MORB Mantle of Workmann & Hart 2005"),
        ("SiO2", 44.7100),
        ("TiO2", 0.1300),
        ("Al2O3", 3.9800),
        ("Fe2O3", 0.1910),
        ("Cr2O3", 0.5700),
        ("FeO", 8.0080),
        ("MnO", 0.1300),
        ("MgO", 38.7300),
        ("NiO", 0.2400),
        ("CaO", 3.1700),
        ("Na2O", 0.2800),
        ("K", 50),
        ("P", 20),
        ("La", 0.192),
        ("Ce", 0.550),
        ("Pb", 0.018),
        ("Pr", 0.107),
        ("Nd", 0.581),
        ("Sr", 7.664),
        ("Zr", 5.082),
        ("Hf", 0.157),
        ("Sm", 0.239),
        ("Eu", 0.096),
        ("Ti", 716.3),
        ("Gd", 0.358),
        ("Tb", 0.070),
        ("Dy", 0.505),
        ("Ho", 0.115),
        ("Y", 3.328),
        ("Er", 0.348),
        ("Yb", 0.365),
        ("Lu", 0.058),
        ("Initial Pressure", 25000.00),
        ("Final Pressure", 10000.00),
        ("Increment Pressure", 0.0),
        ("Initial Temperature", 1450.00),
        ("Final Temperature", 1450.00),
        ("Increment Temperature", 0.0),
        ("Log fO2 Path", "None"),
        ("dp/dt", 0.00),
    ]
).set_index(0, drop=True)[1]

WH_DMM_with_REE = """Title: Depleted MORB Mantle of Workmann & Hart 2005
Initial Composition: SiO2 44.7100
Initial Composition: TiO2 0.1300
Initial Composition: Al2O3 3.9800
Initial Composition: Fe2O3 0.1910
Initial Composition: Cr2O3 0.5700
Initial Composition: FeO 8.0080
Initial Composition: MnO 0.1300
Initial Composition: MgO 38.7300
Initial Composition: NiO 0.2400
Initial Composition: CaO 3.1700
Initial Composition: Na2O 0.2800
Initial Trace: K 50
Initial Trace: P 20
Initial Trace: La 0.192
Initial Trace: Ce 0.550
Initial Trace: Pb 0.018
Initial Trace: Pr 0.107
Initial Trace: Nd 0.581
Initial Trace: Sr 7.664
Initial Trace: Zr 5.082
Initial Trace: Hf 0.157
Initial Trace: Sm 0.239
Initial Trace: Eu 0.096
Initial Trace: Ti 716.3
Initial Trace: Gd 0.358
Initial Trace: Tb 0.070
Initial Trace: Dy 0.505
Initial Trace: Ho 0.115
Initial Trace: Y 3.328
Initial Trace: Er 0.348
Initial Trace: Yb 0.365
Initial Trace: Lu 0.058
Initial Temperature: 1450.00
Final Temperature: 1450.00
Initial Pressure: 25000.00
Final Pressure: 10000.00
Increment Temperature: 0.00
Increment Pressure: 0.00
dp/dt: 0.00
log fo2 Path: None
"""


class Test2MELTSFiles(unittest.TestCase):
    def setUp(self):
        self.df = test_df()
        self.df.loc[:, "Title"] = ["Title {}".format(x) for x in self.df.index.values]
        self.ser = test_ser()
        self.ser.loc["Title"] = "Test_title"

    def test_series_to_melts_file(self):
        ret = to_meltsfiles(self.ser)

    def test_df_to_melts_files(self):
        ret = to_meltsfiles(self.df)

    def test_replicate_file(self):
        """
        Test the directed replication of specific meltsfiles.
        """
        for fileobj in [str_as_file(WH_DMM_with_REE), str_as_file(KM0417_RC12)]:
            pass


class TestFromMELTSFiles(unittest.TestCase):
    def setUp(self):
        pass

    def test_from_melts_file(self):
        file = str_as_file(KM0417_RC12)
        out = from_meltsfile(file)

    def test_df_to_melts_files(self):
        pass

    def test_replicate_ser(self):
        """
        Test the directed replication of specific meltsfile dataframes.
        """
        for ser, file in [
            (WH_DMM_with_REE_ser, str_as_file(WH_DMM_with_REE)),
            (KM0417_RC12_ser, str_as_file(KM0417_RC12)),
        ]:
            df = to_numeric(ser.to_frame().T, errors="ignore")
            imported = to_numeric(from_meltsfile(file).to_frame().T, errors="ignore")
            assert set(map(str.lower, imported.columns)) == set(
                map(str.lower, df.columns)
            )
            df.columns = df.columns.map(str.lower)
            imported.columns = imported.columns.map(str.lower)
            tg = pd.concat([df, imported], axis=0, sort=False)
            wheresame = np.zeros(tg.columns.size) > 0
            where_numeric = np.array(
                [np.issubdtype(tg[c].dtype, np.number) for c in tg.columns]
            )

            n_numeric = where_numeric.sum()
            wheresame[where_numeric] = (
                np.isclose(
                    (
                        tg.iloc[0, :].values[where_numeric]
                        - tg.iloc[1, :].values[where_numeric]
                    ).astype(np.float),
                    np.zeros(n_numeric),
                )[0]
                > 0
            )
            wheresame[~where_numeric] = (
                tg.iloc[0, ~where_numeric] == tg.iloc[1, ~where_numeric]
            )

            self.assertTrue(wheresame.all())


if __name__ == "__main__":
    unittest.main()
