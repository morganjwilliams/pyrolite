from ..util.log import Handle

logger = Handle(__name__)


class Site(object):
    def __init__(self, name=None, coordination=0, affinities={}, mode="cation"):
        """
        Class for specifying mineral sites, including coordination information.

        Will be used for partitioning and affinity calculations for estimating mineral
        site chemistry.
        """

        if name is None:
            name = self.__class__.__name__
        assert mode in ["cation", "anion", "oxygen"]
        self.name = name
        self.coordination = coordination
        self.affinities = affinities
        self.occupancy = None
        self.anionic = mode == "anion"
        self.cationic = mode == "cation"
        self.oxygen = mode == "oxygen"

    def __str__(self):
        """Get a string representation of the site."""
        if self.coordination:
            return """[{}]{}""".format(self.name, self.coordination)
        else:
            return """{}""".format(self.name)

    def __repr__(self):
        """Get a signature of the site."""
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
        super().__init__(name, coordination, *args, mode="cation", **kwargs)


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
        mode="cation",
        **kwargs
    ):
        super().__init__(name, coordination, *args, affinities=affinities, **kwargs)


class IX(Site):
    """
    Dodecahedrally coordinated I site.
    """

    def __init__(self, name="I", coordination=12, *args, **kwargs):
        super().__init__(name, coordination, *args, mode="cation", **kwargs)


class VX(Site):
    """
    Vacancy site.
    """

    def __init__(self, name="V", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, mode="cation", **kwargs)


class OX(Site):
    """
    Oxygen site.
    """

    def __init__(
        self, name="O", coordination=0, affinities={"O{2-}": 0}, *args, **kwargs
    ):
        super().__init__(
            name, coordination, *args, affinities=affinities, mode="oxygen", **kwargs
        )


class AX(Site):
    """
    Anion site.
    """

    def __init__(self, name="A", coordination=0, *args, **kwargs):
        super().__init__(name, coordination, *args, mode="anion", **kwargs)
