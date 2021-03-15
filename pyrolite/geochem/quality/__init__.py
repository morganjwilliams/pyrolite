"""
Submodule for data quality checking and assurance.

Todo
------
    * Identifying interval data by identifying dominant low-end periodicity at scales similar to the lowest values

        This is largely a metadata thing for single points, but a tractable problem for
        multi-point data groups.

        Assumptions around normality or unimodaltiy may not be useful in practice here.

        Non-limited data should exhibit intervals related to the variance
        and overall number of points,

        Could use ratio data here to also include information regarding 'expected value';
        although this is in a way tangential information (the detection limit is  ~
        independent of the data)

        Could use entropy measures either over histograms or over FFT histograms.

        Spectral methods could be useful - its a simlar concept to harmonics in a way;
        offset from a zero-offset to 0+n, 0+2n, 0+3n.. etc peaks with decreasing
        magnitude (which occur at lower value)
"""
from ...util.log import Handle

logger = Handle(__name__)
