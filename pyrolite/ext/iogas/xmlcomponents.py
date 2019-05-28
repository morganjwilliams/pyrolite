import sys
from xml.etree.ElementTree import Element


def Boundary(xpoints, ypoints):
    boundary = Element("Boundary")
    boundary.extend(
        [Element("Point", x=str(x), y=str(y)) for (x, y) in zip(xpoints, ypoints)]
    )
    return boundary


def RegionPolygon(boundary, name="", color=None, description=None):
    c = Element("RegionPolygon", name=name, visible="true")
    sub = []
    if color is not None:
        r, g, b, a = [str(i) for i in color]
        sub.append(Element("Color", r=r, g=g, b=b))
    if description is not None:
        sub.append(Element("Description", name=description))
    sub.append(boundary)
    c.extend(sub)
    return c


def FreeFunctionAxisX(name, function):
    return Element("FreeFunctionAxisX", name=name, function=function)


def FreeFunctionAxisY(name, function):
    return Element("FreeFunctionAxisY", name=name, function=function)


def FreeVariable(name, letter):
    return Element("FreeVariable", letter="A", columnName=name)


def FreeXYDiagram(xvar, yvar):
    diagram = Element("FreeXYDiagram", name="XY Diagram")
    diagram.extend(
        [
            FreeFunctionAxisX(xvar, "a"),
            FreeFunctionAxisY(yvar, "b"),
            FreeVariable(xvar, "A"),
            FreeVariable(xvar, "B"),
        ]
    )
    return diagram


def GeochemFunctionAxisX(name, function, log=False):
    return Element(
        "GeochemFunctionAxisX", name=name, function=function, log=str(log).lower()
    )


def GeochemFunctionAxisY(name, function):
    return Element(
        "GeochemFunctionAxisY", name=name, function=function, log=str(log).lower()
    )


def Variable(element, letter, unit="pct"):
    return Element("Variable", letter="A", element=element, unit=unit)


def GeochemXYDiagram(xvar, yvar):
    diagram = Element("GeochemXYDiagram", name="{} vs. {}".format(yvar, xvar))
    diagram.extend(
        [
            GeochemFunctionAxisX(xvar, "a"),
            GeochemFunctionAxisY(yvar, "b"),
            FreeVariable(xvar, "A"),
            FreeVariable(xvar, "B"),
        ]
    )
    return diagram
