import os, sys
from pathlib import Path
import platform
import pandas as pd
import numpy as np
from .compositions import *
from .util.pd import to_frame, to_numeric
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

RELMASSS_UNITS = {
                  '%': 10**-2,
                  'wt%': 10**-2,
                  'ppm': 10**-6,
                  'ppb': 10**-9,
                  'ppt': 10**-12,
                  'ppq': 10**-15,
                  }


def scale_multiplier(in_unit, target_unit='ppm'):
    """
    Provides the scale difference between to mass units.

    Parameters
    ----------
    in_unit: current units
        Units to be converted from
    target_unit: target mass unit, ppm
        Units to scale to.

    Todo: implement different inputs - string, list, pandas series
    """
    in_unit = str(in_unit).lower()
    target_unit = str(target_unit).lower()
    if not pd.isna(in_unit) and \
        (in_unit in RELMASSS_UNITS.keys()) and \
        (target_unit in RELMASSS_UNITS.keys()):
        scale = RELMASSS_UNITS[in_unit] / RELMASSS_UNITS[target_unit]
    else:
        unkn = [i for i in [in_unit, target_unit] if i not in RELMASSS_UNITS]
        logger.info("Units not known: {}. Defaulting to unity.".format(unkn))
        scale = 1.
    return scale


class RefComp:
    """
    Reference compositional model object, principally used for normalisation.
    """

    def __init__(self, filename, **kwargs):
        self.data = pd.read_csv(filename, **kwargs)
        self.data = self.data.set_index('var')
        self.original_data = self.data.copy() # preserve unaltered record
        #self.add_oxides()
        self.collect_vars()
        self.set_units()
        # self.aggregate_oxides() yet to be implemented

    def aggregate_oxides(self, form='oxide'):
        """
        Compositional models typically include elements in both oxide and
        elemental form, typically divided into 'majors' and 'traces'.

        For the purposes of normalisation - we need
            i) to be able to access values of the form found in the dataset,
            ii) for original values and uncertanties to be preserved, and
            iii) for closure to be preserved.

        There are multiple ways to acheive this - one is to create linked
        element-oxide tables, and another is to force working in one format
        (i.e. Al2O3 (wt%) --> Al (ppm))
        """
        # identify cations to be aggregated

        # for cation in cations:
        #    scale function
        #    aggregate_cation(df: pd.DataFrame, cation, form=form, unit_scale=None)
        raise NotImplementedError('This issue has yet to be addressed.')

    def collect_vars(self,
                     headers=['Reservoir',
                              'Reference',
                              'ModelName',
                              'ModelType'],
                     floatvars=['value',
                                'unc_2sigma',
                                'constraint_value']):
        self.vars = [i for i in self.data.index
                     if (not pd.isna(self.data.loc[i, 'value']))
                         and (i not in headers)]
        self.data.loc[self.vars,
                      floatvars] = to_numeric(self.data.loc[self.vars,
                                                            floatvars],
                                               errors='coerce')

    def set_units(self, to='ppm'):
        v = self.vars
        self.data.loc[v, 'scale'] = \
                            self.data.loc[v, 'units'].apply(scale_multiplier,
                                                            target_unit=to)
        self.data.loc[v, 'units'] = to
        self.data.loc[v, 'value'] = self.data.loc[v, 'value'] * \
                                    self.data.loc[v, 'scale'].astype(np.float)

    def normalize(self, df, aux_cols=["LOD","2SE"]):
        """
        Normalize the values within a dataframe to the refererence composition.
        Here we create indexes for normalisation of values and any auxilary
        values (e.g. uncertainty).

        ## TODO: Implement uncertainty propagation
        """
        dfc = to_frame(df.copy())

        cols = [c for c in dfc.columns if c in self.vars]
        #mdl_ix = cols.copy()
        #df_cols = cols.copy()
        #for c in aux_cols:
        #    df_cols += [v+c for v in cols if v+c in dfc.columns]
        #    mdl_ix += [v for v in cols if v+c in dfc.columns]

        divisor = self.data.loc[cols, 'value'].values
        dfc.loc[:, cols] = np.divide(dfc.loc[:, cols].values,
                                     divisor)
        return dfc

    def ratio(self, ratio):
        """Calculates an elemental ratio."""
        try:
            assert "/" in ratio
            num, den = ratio.split('/')
            return self.data.loc[num, 'value'] / self.data.loc[den, 'value']
        except:
            return np.nan

    def __getattr__(self, var):
        """
        Allow access to model values via attribute e.g. Model.Si
        """
        if not isinstance(var, str):
            var = str(var)
        if var in self.data.index:
            return self.data.loc[var, 'value']
        else:
            return np.nan

    def __getitem__(self, vars):
        """
        Allow access to model values via [] indexing e.g. Model['Si', 'Cr'].
        Currently not implemented for ratios.
        """
        if isinstance(vars, list) or \
           isinstance(vars, pd.Index) or \
           isinstance(vars, np.ndarray):
            vars = [v if isinstance(v, str) else str(v) for v in vars]
        elif not isinstance(vars, str):
            vars = str(vars)
        return self.data.loc[vars, ['value', 'unc_2sigma', 'units']]

    def __repr__(self):
        return "Model of "+self.Reservoir+" ("+self.Reference+")"


def ReferenceCompositions(directory=None, formats=['csv'], **kwargs):
    """
    Build all reference models in a given directory.

    Here we use either the input directory, or the default data directory
    within this module.

    Parameters
    ----------
    directory: file directory, None
        Location of reference data files.
    formats: reference data formats, csv
        List of potential data formats to draw from.
        Currently only csv will work.
    """
    if platform.system() =='Windows':
        kwargs['encoding'] = kwargs.get('encoding', None) or 'cp1252'
    else:
        kwargs['encoding'] = kwargs.get('encoding', None) or 'cp1252'

    curr_dir = os.path.realpath(__file__)
    module_dir = Path(sys.modules['pyrolite'].__file__).parent
    directory = directory or \
                (Path(module_dir) / "data" / "refcomp").resolve()

    assert directory.exists() and directory.is_dir()

    files = []
    for fmt in formats:
        files.extend(directory.glob("./*."+fmt))

    comps = {}
    for f in files:
        r = RefComp(f, **kwargs)
        comps[r.ModelName] = r
    return comps
