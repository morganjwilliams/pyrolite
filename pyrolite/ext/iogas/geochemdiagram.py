from xml.etree.ElementTree import Element
from .common import *


def GeochemFunctionAxisX(name, function, log=False):
    return Element(
        "GeochemFunctionAxisX", name=str(name), function=function, log=str(log).lower()
    )


def GeochemFunctionAxisY(name, function, log=False):
    return Element(
        "GeochemFunctionAxisY", name=str(name), function=function, log=str(log).lower()
    )


def GeochemXYDiagram(xvar, yvar, bounds=None, comments=[], references=[]):
    diagram = Element("GeochemXYDiagram", name="{} vs. {}".format(yvar, xvar))
    diagram.extend(
        [
            GeochemFunctionAxisX(xvar, "a"),
            GeochemFunctionAxisY(yvar, "b"),
            Variable(xvar, "A"),
            Variable(yvar, "B"),
        ]
    )
    if bounds is not None:
        diagram.extend(bounds)
    if comments:
        diagram.extend([Comment(c) for c in comments])
    if references:
        diagram.extend([Reference(r) for r in references])
    return diagram
