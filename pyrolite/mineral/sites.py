from .ions import __default_charges__
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


class Site(object):
    """
    Class for specifying mineral sites, including coordination information.

    Will be used for partitioning and affinity calculations for estimating mineral
    site chemistry.
    """

    def __init__(self, name=None, coordination=0, affinities={}):
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.coordination = coordination
        self.affinities = affinities
        self.occupancy = None
        self.anionic = False
        self.cationic = False
        self.oxygen = False

    def __str__(self):
        if self.coordination:
            return """[{}]{}""".format(self.name, self.coordination)
        else:
            return """{}""".format(self.name)

    def __repr__(self):
        if self.coordination:
            return """{}("{}", {})""".format(
                self.__class__.__name__, self.name, self.coordination
            )
        else:
            return """{}("{}")""".format(self.__class__.__name__, self.name)

    def __eq__(self, other):
        """Check for equality between two sites."""
        # check that the duck quacks/is a Site
        pretest = (
            hasattr(other, "name")
            & hasattr(other, "coordination")
            & hasattr(other, "affinities")
        )
        if pretest:
            # Check for attribute equalilty
            conds = (
                (self.__class__.__name__ == other.__class__.__name__)
                & (self.name == other.name)
                & (self.coordination == other.coordination)
                & (self.affinities == other.affinities)
            )

            return conds
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__().encode("UTF-8"))


class MX(Site):
    """
    Octahedrally coordinated M site.
    """

    def __init__(self, name="M", coordination=8, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)
        self.cationic = True


class TX(Site):
    """
    Tetrahedrally coordinated T site.
    """

    def __init__(
        self,
        name="T",
        coordination=4,
        affinities={"Si{4+}": 0, "Al{3+}": 1, "Fe{3+}": 2},
        *args,
        **kwargs
    ):
        super().__init__(name, coordination, *args, **kwargs)
        self.cationic = True
        self.affinities = affinities


class IX(Site):
    """
    Dodecahedrally coordinated I site.
    """

    def __init__(self, name="I", coordination=12, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)
        self.cationic = True


class VX(Site):
    """
    Vacancy site.
    """

    def __init__(self, name="V", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)
        self.cationic = True


class OX(Site):
    """
    Oxygen site.
    """

    def __init__(
        self, name="O", coordination=0, affinities={"O{2-}": 0}, *args, **kwargs
    ):
        super().__init__(name, coordination, *args, **kwargs)
        self.oxygen = True
        self.affinities = affinities


class AX(Site):
    """
    Anion site.
    """

    def __init__(self, name="A", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, **kwargs)
        self.anionic = True
