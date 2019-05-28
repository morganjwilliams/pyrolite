r"""
Methods for generating xml-based plot templates for use in
`IoGas™ <https://reflexnow.com/iogas/>`__ .
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


from .xmldiagrams import contours_to_FreeXYDiagram

__all__ = ["contours_to_FreeXYDiagram"]
