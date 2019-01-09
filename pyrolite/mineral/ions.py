import periodictable as pt
from periodictable.core import Element
from periodictable.formulas import Formula
from collections import defaultdict
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

__default_charges__ = defaultdict(lambda: None)
__default_charges__.update(
    dict(
        H=1,
        Li=1,
        Be=1,
        B=3,
        C=4,
        O=-2,
        F=-1,
        Na=1,
        Mg=2,
        Al=3,
        Si=4,
        P=3,
        Cl=-1,
        K=1,
        Ca=2,
        Sc=3,
        Ti=4,
        V=3,
        Cr=3,
        Mn=2,
        Fe=2,
        Co=2,
        Ni=2,
        Cu=2,
        Zn=2,
        Br=-1,
        Rb=1,
        Sr=2,
        Y=3,
        Zr=4,
        Nb=5,
        Sn=4,
        I=-1,
        Cs=1,
        Ba=2,
        La=3,
        Ce=3,
        Pr=3,
        Nd=3,
        Sm=3,
        Eu=3,
        Gd=3,
        Tb=3,
        Dy=3,
        Ho=3,
        Er=3,
        Tm=3,
        Yb=3,
        Lu=3,
        Hf=4,
        Pb=2,
        Th=4,
        U=4,
    )
)

# Monkey patching for default charges

Element.default_charge = 0
Formula.default_charge = property(
    lambda self: sum([m * a.default_charge for a, m in self.atoms.items()])
)

for el, c in __default_charges__.items():
    getattr(pt, el).default_charge = c
    assert isinstance(getattr(pt, el), Element)
