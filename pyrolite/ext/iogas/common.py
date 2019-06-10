import sys
from xml.etree.ElementTree import Element
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

def get_color(color):
    r, g, b, a = [str(i) for i in color]
    return Element("Color", r=r, g=g, b=b)


def Variable(element, letter, unit="pct"):
    return Element("Variable", letter=str(letter), element=str(element), unit=str(unit))


def FreeVariable(name, letter):
    return Element("FreeVariable", letter="A", columnName=name)


def Bounds(x, y, width=1.0, height=1.0):
    """
    <Bounds x="-0.021973616" y="-0.04849591" w="1.0688399" h="1.1006972" />
    """
    return Element(
        "Bounds",
        x="{:.5f}".format(x),
        y="{:.5f}".format(y),
        w="{:.5f}".format(width),
        h="{:.5f}".format(height),
    )


def Comment(text):
    c = Element("Comment")
    c.text = text
    return c


def Reference(text):
    r = Element("Reference")
    r.text = text
    return r


def Label(name, xy=(0, 0), color=None, labelangle=0.0, visible=True, strfmt="{:.5f}"):
    x, y = xy
    label = Element(
        "Label",
        name=name,
        visible=str(visible).lower(),
        x=strfmt.format(x),
        y=strfmt.format(y),
    )
    if color is not None:
        label.append(get_color(color))
    label.append(Element("LabelAngle", angle=str(labelangle)))
    return label


def PointFeature(
    name,
    xy=(0.0, 0.0),
    pixelradius=5.0,
    color=None,
    labelangle=0.0,
    visible=True,
    strfmt="{:.5f}",
):
    x, y = xy
    pf = Element(
        "PointFeature",
        name=str(name),
        visible=str(visible).lower(),
        x=strfmt.format(x),
        y=strfmt.format(y),
        pixelRadius=strfmt.format(pixelradius),
    )
    if color is not None:
        pf.append(get_color(color))
    pf.append(Element("LabelAngle", angle=str(labelangle)))
    return pf


def Polygon(xpoints, ypoints, name="", visible=True, strfmt="{:.5f}"):
    """
    Polygon defined by point verticies.
    """
    polygon = Element("Polygon", name=str(name), visible=str(visible).lower())
    polygon.extend(
        [
            Element("Point", x=strfmt.format(x), y=strfmt.format(y))
            for x, y in zip(xpoints, ypoints)
        ]
    )
    return polygon


def Boundary(xpoints, ypoints):
    """
    Boundary polygon defined by point verticies.
    """
    boundary = Element("Boundary")
    boundary.extend(
        [
            Element("Point", x="{:.5f}".format(x), y="{:.5f}".format(y))
            for (x, y) in zip(xpoints, ypoints)
        ]
    )
    return boundary


def BiezerPoint(x, y, sectionEnd=False, strfmt="{:.5f}"):
    """
    Biezer Curve Control point.

    Parameters
    ----------
    x, y : :class:`float`
        Location of the control point.
    sectionEnd : :class:`bool`, :code:`False`
        Whether the control point is an end point (for non-closed paths).
    strfmt : :class:`str`
        Float formatting string.

    Notes
    ------

        * Line segments which have only two points have <sectionEnd="true>
    """
    return Element(
        "BezierPoint",
        x=strfmt.format(x),
        y=strfmt.format(y),
        sectionEnd=str(sectionEnd).lower(),
    )


def Boundary3(xpoints, ypoints, sectionend=False, strfmt="{:.5f}"):
    """
    Boundary defined by segments.

    Parameters
    ----------
    xpoints, ypoints : :class:`numpy.ndarray`
        Location of the control points.


    """
    boundary3 = Element("Boundary3")
    segs = [Element("Linear") for (x, y) in zip(xpoints, ypoints)]
    for ix, s in enumerate(segs):
        s.append(BiezerPoint(xpoints[ix], ypoints[ix], strfmt=strfmt))
    boundary3.extend(segs)
    return boundary3


def Poly(
    name,
    boundary3,
    labelpos=(0.0, 0.0),
    labelangle=0.0,
    color=None,
    closed=True,
    visible=True,
    strfmt="{:.5f}",
):
    poly = Element(
        "Poly", name=name, visible=str(visible).lower(), closed=str(closed).lower()
    )
    if color is not None:
        poly.append(get_color(color))
    poly.append(Element("LabelAngle", angle=str(labelangle)))
    lx, ly = labelpos
    poly.append(Element("LabelPos", x=strfmt.format(lx), y=strfmt.format(ly)))
    poly.append(boundary3)
    return poly


def RegionPolygon(boundary, name="", color=None, description=None):
    c = Element("RegionPolygon", name=name, visible="true")
    sub = []
    if color is not None:
        sub.append(get_color(color))
    if description is not None:
        sub.append(Element("Description", name=description))
    sub.append(boundary)
    c.extend(sub)
    return c
