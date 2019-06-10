"""
Parsing and import for Iogas XML templates.
"""
from xml.etree.ElementTree import ElementTree, Element, parse
from pathlib import Path
from ...util.text import prettify_xml
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class IogasXMLObject(object):
    fltfmt = "{:f}"  # arbitrary precision floats

    def __init__(
        self,
        *args,
        name=None,
        xmlelement=None,
        visible=None,
        closed=None,
        text=None,
        **kwargs
    ):
        self.xmlelement = xmlelement
        self.children = []

        if name is not None:
            self.name = name

        # visual, polygon elements
        if visible is not None:
            self.visible = str(visible).upper() == "TRUE"

        if closed is not None:
            self.closed = str(closed).upper() == "TRUE"

        if text is not None:
            self.text = text

        if self.xmlelement is not None:
            # iter parse
            self._parse_from_element()

    def _parse_from_element(self):
        for child in self.xmlelement:
            self.children.append(xml2pyogas(child))

    def _to_xml(self, **kwargs):
        # if self.xmlelement is not None:
        #    return self.xmlelement
        # else:
        # TODO: have some automated way to export elements
        el = Element(self.__class__.__name__, **kwargs)
        if hasattr(self, "text"):
            el.text = self.text
        return el

    def to_xml(self):
        return self._to_xml()


class Diagram(IogasXMLObject):
    def __init__(self, *args, name="", xmlelement=None, **kwargs):
        super().__init__(name=name, xmlelement=xmlelement, **kwargs)

    def export(self, filename, encoding="utf-8", method="xml"):
        et = pyogas2xml(self)
        ElementTree(et).write(filename, method=method, encoding=encoding)
        return prettify_xml(et)


class FreeXYDiagram(Diagram):
    def __init__(self, *args, name="", xmlelement=None, **kwargs):
        super().__init__(name=name, xmlelement=xmlelement, **kwargs)

    def to_xml(self):
        return self._to_xml(name=str(self.name))


class GeochemXYDiagram(Diagram):
    def __init__(self, *args, name="", xmlelement=None, **kwargs):
        super().__init__(name=name, xmlelement=xmlelement, **kwargs)

    def to_xml(self):
        return self._to_xml(name=str(self.name))


class FreeTernaryDiagram(Diagram):
    def __init__(self, *args, name="", xmlelement=None, **kwargs):
        super().__init__(name=name, xmlelement=xmlelement, **kwargs)

    def to_xml(self):
        return self._to_xml(name=str(self.name))


class Axis(IogasXMLObject):
    def __init__(self, *args, name="", log=False, xmlelement=None, **kwargs):
        super().__init__(name=name, xmlelement=xmlelement, **kwargs)
        self.log = log

    def to_xml(self):
        return self._to_xml(name=str(self.name), log=str(self.log))


class FunctionAxis(Axis):
    def __init__(
        self, *args, name="", function="", log=False, xmlelement=None, **kwargs
    ):
        super().__init__(
            name=name, log=log, function="", xmlelement=xmlelement, **kwargs
        )
        self.function = function

    def to_xml(self):
        return self._to_xml(
            name=str(self.name), function=str(self.function), log=str(self.log)
        )


class FreeAxis(Axis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FreeAxisA(FreeAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FreeAxisB(FreeAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FreeAxisC(FreeAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FreeFunctionAxisX(FunctionAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FreeFunctionAxisY(FunctionAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GeochemFunctionAxisX(FunctionAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GeochemFunctionAxisY(FunctionAxis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Variable(IogasXMLObject):
    def __init__(
        self, *args, letter="", element="", unit="", xmlelement=None, **kwargs
    ):
        super().__init__(
            *args,
            letter=letter,
            element=element,
            unit=unit,
            xmlelement=xmlelement,
            **kwargs
        )
        self.letter = letter
        self.element = element
        self.unit = unit

    def to_xml(self):
        return self._to_xml(
            letter=str(self.letter), element=str(self.element), unit=str(self.unit)
        )


class FreeVariable(IogasXMLObject):
    def __init__(self, *args, letter="", columnName="", xmlelement=None, **kwargs):
        super().__init__(
            *args, letter=letter, columnName=columnName, xmlelement=xmlelement, **kwargs
        )
        self.letter = letter
        self.columnName = columnName

    def to_xml(self):
        return self._to_xml(letter=str(self.letter), columnName=str(self.columnName))


class Description(IogasXMLObject):
    def __init__(self, *args, name="", xmlelement=None, **kwargs):
        super().__init__(xmlelement=xmlelement, name=name, **kwargs)

    def to_xml(self):
        return self._to_xml(name=str(self.name))


class TextField(IogasXMLObject):
    def __init__(self, *args, text=None, xmlelement=None, **kwargs):
        super().__init__(xmlelement=xmlelement, text=text or "", **kwargs)


class Reference(TextField):
    def __init__(self, *args, text=None, xmlelement=None, **kwargs):
        super().__init__(xmlelement=xmlelement, text=text, **kwargs)


class Comment(TextField):
    def __init__(self, *args, text=None, xmlelement=None, **kwargs):
        super().__init__(xmlelement=xmlelement, text=text, **kwargs)

    def to_xml(self):
        return self._to_xml()


class LabelAngle(IogasXMLObject):
    def __init__(self, *args, angle=0.0, xmlelement=None, **kwargs):
        super().__init__(xmlelement=xmlelement, **kwargs)
        self.angle = float(angle)

    def to_xml(self):
        return self._to_xml(angle=self.fltfmt.format(self.angle))


class LabelPos(IogasXMLObject):
    def __init__(
        self, *args, x=None, y=None, a=None, b=None, xmlelement=None, **kwargs
    ):
        super().__init__(xmlelement=xmlelement, **kwargs)
        if x is not None:
            self.x = float(x)
            self.y = float(y)
        if a is not None:
            self.a = float(a)
            self.b = float(b)

    def to_xml(self):
        if hasattr(self, "x"):  # cartesian mode
            return self._to_xml(
                x=self.fltfmt.format(self.x), y=self.fltfmt.format(self.y)
            )
        else:  # ternary mode
            return self._to_xml(
                a=self.fltfmt.format(self.a), b=self.fltfmt.format(self.b)
            )


class Label(IogasXMLObject):
    def __init__(
        self, *args, name="", x=0.0, y=0.0, visible=True, xmlelement=None, **kwargs
    ):
        super().__init__(xmlelement=xmlelement, **kwargs)
        self.name = name
        self.x = float(x)
        self.y = float(y)


class Colour(IogasXMLObject):
    def __init__(self, *args, r=255, g=255, b=255, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)

    def to_xml(self):
        return self._to_xml(r=str(self.r), g=str(self.g), b=str(self.b))


class Point(IogasXMLObject):
    def __init__(self, *args, x=0.0, y=0.0, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)
        self.x = float(x)
        self.y = float(y)

    def to_xml(self):
        return self._to_xml(x=self.fltfmt.format(self.x), y=self.fltfmt.format(self.y))


class BezierPoint(IogasXMLObject):
    def __init__(
        self, *args, x=0.0, y=0.0, sectionEnd=False, xmlelement=None, **kwargs
    ):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)
        self.x = float(x)
        self.y = float(y)
        self.sectionEnd = str(sectionEnd).upper() == "TRUE"

    def to_xml(self):
        return self._to_xml(
            x=self.fltfmt.format(self.x),
            y=self.fltfmt.format(self.y),
            sectionEnd=str(self.sectionEnd).lower(),
        )


class TPoint(IogasXMLObject):
    def __init__(self, *args, a=0.0, b=0.0, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)
        self.a = float(a)
        self.b = float(b)

    def to_xml(self):
        return self._to_xml(a=self.fltfmt.format(self.a), b=self.fltfmt.format(self.b))


class PointFeature(IogasXMLObject):
    def __init__(
        self,
        *args,
        x=0.0,
        y=0.0,
        pixelRadius=5.0,
        visible=True,
        xmlelement=None,
        **kwargs
    ):
        super().__init__(*args, visible=visible, xmlelement=xmlelement, **kwargs)
        self.x = float(x)
        self.y = float(y)
        self.pixelRadius = float(pixelRadius)

    def to_xml(self):
        return self._to_xml(
            name=str(self.name),
            x=self.fltfmt.format(self.x),
            y=self.fltfmt.format(self.y),
            pixelRadius=self.fltfmt.format(self.pixelRadius),
            visible=str(self.visible).lower(),
        )


class Linear(IogasXMLObject):
    def __init__(self, *args, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)


class Boundary3(IogasXMLObject):
    def __init__(self, *args, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)


class Boundary(IogasXMLObject):
    def __init__(self, *args, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)


class Bounds(IogasXMLObject):
    def __init__(self, *args, x=0.0, y=0.0, w=0.0, h=0.0, xmlelement=None, **kwargs):
        super().__init__(*args, xmlelement=xmlelement, **kwargs)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    def to_xml(self):
        return self._to_xml(
            x=self.fltfmt.format(self.x),
            y=self.fltfmt.format(self.y),
            w=self.fltfmt.format(self.w),
            h=self.fltfmt.format(self.h),
        )


class RegionPolygon(IogasXMLObject):
    def __init__(self, *args, name="", visible=True, xmlelement=None, **kwargs):
        super().__init__(name=name, visible=visible, xmlelement=xmlelement, **kwargs)

    def to_xml(self):
        return self._to_xml(name=str(self.name), visible=str(self.visible).lower())


class Poly(IogasXMLObject):
    def __init__(
        self,
        *args,
        name="",
        function="",
        visible=True,
        closed=True,
        xmlelement=None,
        **kwargs
    ):
        super().__init__(
            name=name, closed=closed, visible=visible, xmlelement=xmlelement, **kwargs
        )

    def to_xml(self):
        return self._to_xml(
            name=str(self.name),
            visible=str(self.visible).lower(),
            closed=str(self.closed).lower(),
        )


class Polygon(IogasXMLObject):
    def __init__(self, name="", *args, xmlelement=None, visible=True, **kwargs):
        super().__init__(name=name, xmlelement=xmlelement, visible=visible, **kwargs)

    def to_xml(self):
        return self._to_xml(name=str(self.name), visible=str(self.visible).lower())


# mapping between xml elements and python classes
__mapping__ = {
    "GeochemXYDiagram": GeochemXYDiagram,
    "FreeXYDiagram": FreeXYDiagram,
    "FreeTernaryDiagram": FreeTernaryDiagram,
    "GeochemFunctionAxisX": GeochemFunctionAxisX,
    "GeochemFunctionAxisY": GeochemFunctionAxisY,
    "FreeFunctionAxisX": FreeFunctionAxisX,
    "FreeFunctionAxisY": FreeFunctionAxisY,
    "FreeAxisA": FreeAxisA,
    "FreeAxisB": FreeAxisB,
    "FreeAxisC": FreeAxisC,
    "Variable": Variable,
    "Poly": Poly,
    "Reference": Reference,
    "Comment": Comment,
    "Label": Label,
    "Colour": Colour,
    "LabelAngle": LabelAngle,
    "LabelPos": LabelPos,
    "Boundary3": Boundary3,
    "Boundary": Boundary,
    "Linear": Linear,
    "BezierPoint": BezierPoint,
    "Polygon": Polygon,
    "RegionPolygon": RegionPolygon,
    "Point": Point,
    "TPoint": TPoint,
    "Bounds": Bounds,
    "PointFeature": PointFeature,
    "FreeVariable": FreeVariable,
    "Description": Description,
}


def xml2pyogas(xmlelement):
    """
    Return a python class representation of an Iogas template xml element.

    Parameters
    -----------
    xmlelement : :class:`xml.etree.ElementTree.Element`
        Element to convert a class representation.
    """

    if xmlelement.tag not in __mapping__:
        logger.warning("XML Element not Known")
    cls = __mapping__[xmlelement.tag]
    return cls(xmlelement=xmlelement, text=xmlelement.text, **xmlelement.attrib)


def pyogas2xml(el):
    """
    Create an xml document from a class representation of Iogas xml elements.

    Parameters
    --------------
    el : :class:`IogasXMLObject`
    """
    elxml = el.to_xml()
    if el.children:
        c = [pyogas2xml(i) for i in el.children]
        elxml.extend(c)
    return elxml


def import_iogas_diagram(filepath):
    tree = parse(filepath)
    root = tree.getroot()
    form = root.tag
    assert form in ["GeochemXYDiagram", "FreeXYDiagram", "FreeTernaryDiagram"]
    return xml2pyogas(root)
