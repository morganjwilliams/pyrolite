"""
Geological Timescale
======================

pyrolite includes a simple geological timescale, based on a recent verion
of the International Chronostratigraphic Chart [#ICS]_. The
:class:`~pyrolite.util.time.Timescale` class can be used to look up names for
specific geological ages, to look up times for known geological age names
and to access a reference table for all of these.

.. [#ICS] Cohen, K.M., Finney, S.C., Gibbard, P.L., Fan, J.-X., 2013.
    `The ICS International Chronostratigraphic Chart <http://www.stratigraphy.org/index.php/ics-chart-timescale>`__.
    Episodes 36, 199–204.
"""
########################################################################################
# First we'll create a timescale:
#
from pyrolite.util.time import Timescale

ts = Timescale()
########################################################################################
# From this we can look up the names of ages (in million years, or Ma):
#
ts.named_age(1212.1)
########################################################################################
# As geological age names are hierarchical, the name you give an age depends on what
# level you're looking at. By default, the timescale will return the most specific
# non-null level. The levels accessible within the timescale are listed
# as an attribute:
#
ts.levels
########################################################################################
# These can be used to refine the output names to your desired level of specificity
# (noting that for some ages, the levels which are accessible can differ; see the chart):
#
ts.named_age(1212.1, level="Epoch")
########################################################################################
# The timescale can also do the inverse for you, and return the timing information for a
# given named age:
ts.text2age("Holocene")
#########################################################################################
# We can use this to create a simple template to visualise the geological timescale
# (noting that the the official colours have not yet been implemented):
#
import matplotlib.pyplot as plt

df = ts.data
fig, ax = plt.subplots(1, figsize=(5, 10))

for ix, level in enumerate(ts.levels):
    ldf = df.loc[df.Level == level, :]
    for r in ldf.index:
        rdf = ldf.loc[r, :]
        duration = rdf.Start - rdf.End
        ax.bar(ix, duration, bottom=rdf.End, width=1, edgecolor="k")

ax.set_xticks(range(len(ts.levels)))
ax.set_xticklabels(ts.levels, rotation=60)
ax.xaxis.set_ticks_position("top")
ax.set_ylabel("Age (Ma)")
ax.invert_yaxis()
