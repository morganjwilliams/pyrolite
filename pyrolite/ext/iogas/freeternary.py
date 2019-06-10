from xml.etree.ElementTree import Element
from .common import *


def TPoint(a, b, c=None, strfmt="{:.5f}"):
    """
    Ternary point defined by a and b axes.
    """
    return Element("TPoint", a=strfmt.format(a), b=strfmt.format(b))


def FreeAxisA(name, log=False):
    return Element("FreeAxisA", name=str(name), log=str(log).lower())


def FreeAxisB(name, log=False):
    return Element("FreeAxisB", name=str(name), log=str(log).lower())


def FreeAxisC(name, log=False):
    return Element("FreeAxisC", name=str(name), log=str(log).lower())


def FreeTernaryDiagram(name, avar, bvar, cvar, bounds=None, comments=[], references=[]):
    diagram = Element(
        "FreeTernaryDiagram", name="{} - {} - {}".format(avar, bvar, cvar)
    )
    diagram.extend([FreeAxisA(avar), FreeAxisB(bvar), FreeAxisC(cvar)])
    if bounds is not None:
        diagram.extend(bounds)
    if comments:
        diagram.extend([Comment(c) for c in comments])
    if references:
        diagram.extend([Reference(r) for r in references])
    return diagram
