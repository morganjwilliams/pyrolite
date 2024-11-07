"""
Creating  Plot Templates/Classifier Models
==========================================

:mod:`pyrolite` provides a system for creating and using plot templates/classifier models
based on a series of polygons in variable space (e.g., the TAS diagram). A variety of
`diagram templates <../examples/plotting/templates.html>`__/ `classifiers <../examples/util/TAS.html>`__
are available, but you can also create your own.

In this tutorial, we'll go through the process of creating a diagram template from scratch,
as a demonstration of how you might create your own for your use - or to later contribute
to the collection in pyrolite.
"""

#######################################################################################
# The basis for most diagrams and classifiers is the class
# :class:`~pyrolite.util.classification.PolygonClassifier`; the docstring-based help
# text is a good place to start to understand what we'll need to put it together:
#
from pyrolite.util.classification import PolygonClassifier

help(PolygonClassifier)
#######################################################################################
# The key things you'll need to construct a classifer are:
#
# 1. a name
# 2. a specification of what the axes correspond to,
# 3. and a dictionary of fields (dictionaries containing a 'name' and coordinates defining the polygon).
#
# We can also optionally specify the x and y limits, which are specific to plotting.
# Here we'll put together a simple classifier model with just two fields,
# and add this to a :mod:`matplotlib` axis. You can optionally specify names/labels for each field, \
# here we opt to just use some basic IDs (A and B), so these are what will be added to the plot:
#

clf = PolygonClassifier(
    name="DemoClassifier",
    axes={"x": "SiO2", "y": "MgO"},
    fields={
        "A": {
            "poly": [[0, 75], [0, 100], [50, 100], [50, 75]],
        },
        "B": {
            "poly": [[0, 25], [0, 75], [25, 75], [25, 25]],
        },
    },
    xlim=(0, 100),
    ylim=(0, 100),
)
ax = clf.add_to_axes(add_labels=True)
ax.figure
#######################################################################################
# While we're individually passing each of these arguments to
# :class:`~pyrolite.util.classification.PolygonClassifier`, we can also pass a dictionary
# of keyword arguments:
#
cfg = dict(
    name="DemoClassifier",
    axes={"x": "SiO2", "y": "MgO"},
    fields={
        "A": {
            "poly": [[0, 75], [0, 100], [50, 100], [50, 75]],
        },
        "B": {
            "poly": [[0, 25], [0, 75], [25, 75], [25, 25]],
        },
    },
    xlim=(0, 100),
    ylim=(0, 100),
)

clf = PolygonClassifier(**cfg)
#######################################################################################
# Each of the built-in models are saved as JSON files, and loaded in a manner as above;
# we can replicate that here - saving our configuration to JSON then loading it up again.
# We'll use a temporary directory here, but you can save it wherever you like (note the
# :mod:`pyrolite` templates live under `/data/models` in the repository); once you've
# got a template working how you'd like, consider
# `submitting it <../../dev/contributing.html>`__!
#
import json

from pyrolite.util.general import temp_path

tmp = temp_path()
with open(tmp / "demo_classifier.json", "w") as f:
    f.write(json.dumps(cfg))


with open(tmp / "demo_classifier.json", "r") as f:
    clf = PolygonClassifier(**json.load(f))

clf.add_to_axes(add_labels=True).figure
#######################################################################################
# Ternary Templates
# ~~~~~~~~~~~~~~~~~
#
# While it's slightly more work, you can also generate ternary templates using a very
# simliar pattern to the bivariate ones above. The principal differences are that you'll 
# need to specify three axes (t, l, r), specify a 'ternary' transform, and have coordinates
# for polygons in the ternary space - each with three values. For example, 
# here are two fields from the UDSA soil texture triangle:
#
cfg = {
    "axes": {"t": "Clay", "l": "Sand", "r": "Silt"},
    "transform": "ternary",
    "fields": {
        "sand": {"name": "Sand", "poly": [[0, 100, 0], [10, 90, 0], [0, 85, 15]]},
        "loamy-sand": {
            "name": "Loamy Sand",
            "poly": [[10, 90, 0], [0, 85, 15], [0, 70, 30], [15, 85, 0]],
        },
    },
}
PolygonClassifier(**cfg).add_to_axes(add_labels=True).figure