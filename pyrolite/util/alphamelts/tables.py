import os, sys
import re
import io
import logging
import pandas as pd
from pathlib import Path
from pyrolite.util.pd import to_frame, to_ser
from pyrolite.geochem.ind import __common_elements__, __common_oxides__
from .parse import from_melts_cstr
from .meltsfile import to_meltsfiles

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class MeltsOutput(object):
    def __init__(self, directory, kelvin=True):
        self.title = None
        self.kelvin = kelvin
        self.phasenames = set([])
        self.majors = set([])
        self.traces = set([])
        self.phases = {}
        dir = Path(directory)
        for name, table, load in [
            ("bulkcomp", "Bulk_comp_tbl.txt", self._read_bulkcomp),
            ("solidcomp", "Solid_comp_tbl.txt", self._read_solidcomp),
            ("liquidcomp", "Liquid_comp_tbl.txt", self._read_liquidcomp),
            ("phasemass", "Phase_mass_tbl.txt", self._read_phasemass),
            ("phasevol", "Phase_vol_tbl.txt", self._read_phasevol),
            ("tracecomp", "Trace_main_tbl.txt", self._read_trace),
            ("system", "System_main_tbl.txt", self._read_systemmain),
            ("phasemain", "Phase_main_tbl.txt", self._read_phasemain),
        ]:
            tpath = dir / table
            setattr(self, name, load(tpath))
            # logger.warning("Error on table import: {}".format(tpath))

    @property
    def tables(self):
        """
        Get the set of tables accesible from the output object.

        Returns
        -------
        :class:`set`
            Tables accesible from the :class:`MeltsOutput` object.
        """
        return {
            "bulkcomp",
            "solidcomp",
            "liquidcomp",
            "phasemass",
            "phasevol",
            "tracecomp",
            "system",
        }

    def _set_title(self, title):
        if self.title is None:
            self.title = title
        else:
            if title == self.title:
                pass
            else:
                logger.warning(
                    "File with conflicting title found: {}; expected {}".format(
                        title, self.title
                    )
                )

    def _get_table_title(self, filepath):
        with open(filepath) as f:
            first_line = f.readline()
            title = first_line.replace("Title: ", "").strip()
            return title

    def read_table(self, filepath, **kwargs):
        """
        Read a melts table (a space-separated value file).

        Parameters
        -----------
        filepath : :class:`str` | :class:`pathlib.Path`
            Filepath to the melts table.

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame with table information.
        """
        path = Path(filepath)
        if path.exists and path.is_file:
            self._set_title(self._get_table_title(filepath))
            df = pd.read_csv(filepath, sep=" ", **kwargs)
            df = df.dropna(how="all", axis=1)
            if ("Temperature" in df.columns) and not self.kelvin:
                df.Temperature -= 273.15
            return df
        else:
            logger.warning("Expected file {} does not exist.".format(filepath))

    def _read_solidcomp(self, filepath, skiprows=3):
        table = self.read_table(filepath, skiprows=skiprows)
        return table

    def _read_liquidcomp(self, filepath, skiprows=3):
        table = self.read_table(filepath, skiprows=skiprows)
        return table

    def _read_bulkcomp(self, filepath, skiprows=3):
        table = self.read_table(filepath, skiprows=skiprows)
        return table

    def _read_phasemass(self, filepath, skiprows=3):
        table = self.read_table(filepath, skiprows=skiprows)
        self.phasenames = self.phasenames | set(
            [
                i
                for i in table.columns
                if i not in ["Pressure", "Temperature", "mass", "V"]
            ]
        )
        return table

    def _read_phasevol(self, filepath, skiprows=3):
        table = self.read_table(filepath, skiprows=skiprows)
        self.phasenames = self.phasenames | set(
            [
                i
                for i in table.columns
                if i not in ["Pressure", "Temperature", "mass", "V"]
            ]
        )
        return table

    def _read_systemmain(self, filepath, skiprows=3):
        table = self.read_table(filepath, skiprows=skiprows)
        return table

    def _read_trace(self, filepath, header=3):
        pass

    def _read_phasemain(self, filepath, header=3):
        # can throw errors if run before alphamelts is exited
        with open(filepath) as f:
            data = f.read().split("\n\n")[1:]
            for tab in data:
                lines = re.split(r"[\n\r]", tab)
                phase = lines[0].split()[0].strip()
                table = pd.read_csv(
                    io.BytesIO("\n".join(lines[1:]).encode("UTF-8")), sep=" "
                )
                if "formula" in table.columns:
                    table.loc[:, "formula"] = table.loc[:, "formula"].apply(
                        from_melts_cstr
                    )
                if ("Temperature" in table.columns) and not self.kelvin:
                    table.Temperature -= 273.15
                self.phases[phase] = table

    def _read_logfile(filepath):
        pass
