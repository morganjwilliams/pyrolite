import sys
from xml.etree.ElementTree import Element
from .common import *
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

def FreeFunctionAxisX(name, function):
    return Element("FreeFunctionAxisX", name=name, function=function)


def FreeFunctionAxisY(name, function):
    return Element("FreeFunctionAxisY", name=name, function=function)


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
