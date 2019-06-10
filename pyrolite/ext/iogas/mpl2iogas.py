import numpy as np
from xml.etree.ElementTree import ElementTree
from . import freediagram
from . import geochemdiagram
from ...util.text import int_to_alpha, prettify_xml
from ...util.plot import get_contour_paths


def contours_to_FreeXYDiagram(
    ax,
    xvar="X",
    yvar="Y",
    filename="element.xml",
    contournames=None,
    resolution=100,
    description_prefix="",
    encoding="utf-8",
):
    """
    Take the contour lines from an axis and convert them to an iogas xml diagram
    template.

    Parameters
    ------------

    Note
    ------

        The polygons need not return to the same point.
    """
    diagram = freediagram.FreeXYDiagram(xvar, yvar)
    cpaths, cnames, styles = get_contour_paths(ax, resolution=resolution)
    if contournames is not None:
        assert len(contournames) == len(cpaths)
        cnames = contournames
    # create contours
    contours = []
    for ix, (p, name, sty) in enumerate(zip(cpaths, cnames, styles)):
        for six, subpath in enumerate(p):
            if len(p) != 1:
                suffix = "-" + int_to_alpha(six)
            else:
                suffix = ""
            cname = ["Countour-{}".format(name), "Countour-{}".format(ix)][
                name is None
            ] + suffix
            c = freediagram.RegionPolygon(
                freediagram.Boundary(*subpath),
                color=sty["color"],
                name=cname,
                description=description_prefix,
            )
        contours.append(c)
    diagram.extend(contours)
    ElementTree(diagram).write(filename, method="xml", encoding="utf-8")
    return prettify_xml(diagram)


def contours_to_GeochemXYDiagram(
    ax,
    xvar="X",
    yvar="Y",
    filename="element.xml",
    contournames=None,
    resolution=100,
    description_prefix="",
    encoding="utf-8",
):
    """
    Take the contour lines from an axis and convert them to an iogas xml diagram
    template.

    Parameters
    ------------

    Note
    ------

        The polygons need not return to the same point.
    """
    diagram = geochemdiagram.GeochemXYDiagram(xvar, yvar)
    cpaths, cnames, styles = get_contour_paths(ax, resolution=resolution)
    if contournames is not None:
        assert len(contournames) == len(cpaths)
        cnames = contournames
    # create contours
    contours = []
    for ix, (p, name, sty) in enumerate(zip(cpaths, cnames, styles)):
        for six, subpath in enumerate(p):
            if len(p) != 1:
                suffix = "-" + int_to_alpha(six)
            else:
                suffix = ""
            cname = ["Countour-{}".format(name), "Countour-{}".format(ix)][
                name is None
            ] + suffix
            c = geochemdiagram.Poly(
                str(name), geochemdiagram.Boundary3(*subpath), color=sty["color"]
            )
            # boundary

        contours.append(c)
    diagram.extend(contours)
    ElementTree(diagram).write(filename, method="xml", encoding="utf-8")
    return prettify_xml(diagram)
