"""
Geological Timescale
=====================

pyrolite includes a simple geological timescale, based on a recent verion
of the International Chronostratigraphic Chart [#ICS]_. The
:class:`~pyrolite.util.time.Timescale` class can be used to look up names for
specific geological ages, to look up times for known geological age names
and to access a reference table for all of these.
"""
from pyrolite.util.time import Timescale, age_name

ts = Timescale()

eg = ts.data.iloc[:, :5]  # the first five columns of this data table
eg
########################################################################################
# References
# ~~~~~~~~~~~
#
# .. [#ICS] Cohen, K.M., Finney, S.C., Gibbard, P.L., Fan, J.-X., 2013.
#     `The ICS International Chronostratigraphic Chart <http://www.stratigraphy.org/index.php/ics-chart-timescale>`__.
#     Episodes 36, 199–204.
#
# .. seealso::
#
#   Examples:
#     `Timescale <../examples/util/timescale.html>`__
#
#   Modules, Classes and Functions:
#     :mod:`pyrolite.util.time`,
#     :class:`~pyrolite.util.time.Timescale`
