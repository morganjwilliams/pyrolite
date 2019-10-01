"""
Submodule for working with geochemical data.
"""
import logging
import pandas as pd

logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# direct imports for backwards compatibility
from .ind import *
from .magma import *
from .parse import *
from .transform import *
from .validate import *
from .alteration import *
from .norm import *

# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrochem")
@pd.api.extensions.register_dataframe_accessor("pyrochem")
class pyrochem(object):
    """
    Custom dataframe accessor for pyrolite geochemistry.
    """

    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    # pyrolite.geochem.parse functions

    def check_multiple_cation_inclusion(self, exclude=["LOI", "FeOT", "Fe2O3T"]):
        """
        Returns cations which are present in both oxide and elemental form.

        Parameters
        -----------
        exclude : :class:`list`, :code:`["LOI", "FeOT", "Fe2O3T"]`
            List of components to exclude from the duplication check.

        Returns
        --------
        :class:`set`
            Set of elements for which multiple components exist in the dataframe.

        Todo
        -----
            * Options for output (string/formula).
        """

        obj = self._obj
        return check_multiple_cation_inclusion(obj, exclude=exclude)

    # pyrolite.geochem.transform functions

    def to_molecular(self, renorm=True):
        """
        Converts mass quantities to molar quantities of the same order.

        Parameters
        -----------
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after converting to relative moles.

        Note
        ------
        Does not convert units (i.e. mass% --> mol%; mass-ppm --> mol-ppm).

        Returns
        -------
        :class:`pandas.DataFrame`
            Transformed dataframe.
        """
        obj = self._obj
        return to_molecular(obj, renorm=renorm)

    def to_weight(self, renorm=True):
        """
        Converts molar quantities to mass quantities of the same order.

        Parameters
        -----------
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after converting to relative moles.

        Note
        ------
        Does not convert units (i.e. mol% --> mass%; mol-ppm --> mass-ppm).

        Returns
        -------
        :class:`pandas.DataFrame`
            Transformed dataframe.
        """
        obj = self._obj
        return to_weight(obj, renorm=renorm)

    def devolatilise(
        self, exclude=["H2O", "H2O_PLUS", "H2O_MINUS", "CO2", "LOI"], renorm=True
    ):
        """
        Recalculates components after exclusion of volatile phases (e.g. H2O, CO2).

        Parameters
        -----------
        exclude : :class:`list`
            Components to exclude from the dataset.
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after devolatilisation.

        Returns
        -------
        :class:`pandas.DataFrame`
            Transformed dataframe.
        """
        obj = self._obj
        return devolatilise(obj, exclude=exclude, renorm=renorm)

    def elemental_sum(
        self, component=None, to=None, total_suffix="T", logdata=False, molecular=False
    ):
        """
        Sums abundance for a cation to a single series, starting from a
        dataframe containing multiple componnents with a single set of units.

        Parameters
        ----------
        component : :class:`str`
            Component indicating which element to aggregate.
        to : :class:`str`
            Component to cast the output as.
        logdata : :class:`bool`, :code:`False`
            Whether data has been log transformed.
        molecular : :class:`bool`, :code:`False`
            Whether to perform a sum of molecular data.

        Returns
        -------
        :class:`pandas.Series`
            Series with cation aggregated.
        """
        obj = self._obj
        return elemental_sum(
            obj,
            component=component,
            to=to,
            total_suffix=total_suffix,
            logdata=logdata,
            molecular=molecular,
        )

    def aggregate_element(
        self, to, total_suffix="T", logdata=False, renorm=False, molecular=False
    ):
        """
        Aggregates cation information from oxide and elemental components to either a
        single species or a designated mixture of species.

        Parameters
        ----------
        to : :class:`str` | :class:`~periodictable.core.Element` | :class:`~periodictable.formulas.Formula`  | :class:`dict`
            Component(s) to convert to. If one component is specified, the element will be
            converted to the target species.

            If more than one component is specified with proportions in a dictionary
            (e.g. :code:`{'FeO': 0.9, 'Fe2O3': 0.1}`), the components will be split as a
            fraction of the elemental sum.
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after recalculation.
        total_suffix : :class:`str`, 'T'
            Suffix of 'total' variables. E.g. 'T' for FeOT, Fe2O3T.
        logdata : :class:`bool`, :code:`False`
            Whether the data has been log transformed.
        molecular : :class:`bool`, :code:`False`
            Whether to perform a sum of molecular data.

        Notes
        -------
        This won't convert units, so need to start from single set of units.

        Returns
        -------
        :class:`pandas.Series`
            Series with cation aggregated.
        """
        obj = self._obj
        return aggregate_element(
            obj,
            to,
            total_suffix=total_suffix,
            logdata=logdata,
            renorm=renorm,
            molecular=molecular,
        )

    def recalculate_Fe(
        self, to="FeOT", renorm=True, total_suffix="T", logdata=False, molecular=False
    ):
        """
        Recalculates abundances of iron, and normalises a dataframe to contain  either
        a single species, or multiple species in certain proportions.

        Parameters
        -----------
        to : :class:`str` | :class:`~periodictable.core.Element` | :class:`~periodictable.formulas.Formula`  | :class:`dict`
            Component(s) to convert to.

            If one component is specified, all iron will be
            converted to the target species.

            If more than one component is specified with proportions in a dictionary
            (e.g. :code:`{'FeO': 0.9, 'Fe2O3': 0.1}`), the components will be split as a
            fraction of Fe.
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after recalculation.
        total_suffix : :class:`str`, 'T'
            Suffix of 'total' variables. E.g. 'T' for FeOT, Fe2O3T.
        logdata : :class:`bool`, :code:`False`
            Whether the data has been log transformed.
        molecular : :class:`bool`, :code:`False`
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        :class:`pandas.DataFrame`
            Transformed dataframe.
        """

        obj = self._obj
        return recalculate_Fe(
            obj,
            to,
            total_suffix=total_suffix,
            logdata=logdata,
            renorm=renorm,
            molecular=molecular,
        )

    def add_ratio(
        self,
        ratio: str,
        alias: str = "",
        norm_to=None,
        convert=lambda x: x,
        molecular=False,
    ):
        """
        Add a ratio of components A and B, given in the form of string 'A/B'.
        Returned series be assigned an alias name.

        Parameters
        -----------
        df : :class:`pandas.DataFrame`
            Dataframe to append ratio to.
        ratio : :class:`str`
            String decription of ratio in the form A/B[_n].
        alias : :class:`str`
            Alternate name for ratio to be used as column name.
        norm_to : :class:`str` | :class:`pyrolite.geochem.norm.RefComp`, `None`
            Reference composition to normalise to.
        molecular : :class:`bool`, :code:`False`
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with ratio appended.

        Todo
        ------
            * Use elemental sum from reference compositions
            * Use sympy-like functionality to accept arbitrary input e.g.
                :code:`"MgNo = Mg / (Mg + Fe)"` for subsequent calculation.

        See Also
        --------
        :func:`~pyrolite.geochem.transform.add_MgNo`
        """
        obj = self._obj
        return add_ratio(
            obj, ratio, alias, norm_to=norm_to, convert=convert, molecular=molecular
        )

    def add_MgNo(
        self, molecular=False, use_total_approx=False, approx_Fe203_frac=0.1, name="Mg#"
    ):
        """
        Append the magnesium number to a dataframe.

        Parameters
        ----------
        molecular : :class:`bool`, :code:`False`
            Whether the input data is molecular.
        use_total_approx : :class:`bool`, :code:`False`
            Whether to use an approximate calculation using total iron rather than just FeO.
        approx_Fe203_frac : :class:`float`
            Fraction of iron which is oxidised, used in approximation mentioned above.
        name : :class:`str`
            Name to use for the Mg Number column.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with ratio appended.

        See Also
        --------
        :func:`~pyrolite.geochem.transform.add_ratio`
        """

        obj = self._obj
        return add_MgNo(
            obj,
            molecular=molecular,
            use_total_approx=use_total_approx,
            approx_Fe203_frac=approx_Fe203_frac,
            name=name,
        )

    @update_docstring_references
    def lambda_lnREE(
        self,
        norm_to="Chondrite_PON",
        exclude=["Pm", "Eu"],
        params=None,
        degree=4,
        append=[],
        scale="ppm",
        **kwargs
    ):
        """
        Calculates orthogonal polynomial coefficients (lambdas) for a given set of REE data,
        normalised to a specific composition [#ref_1]_. Lambda factors are given for the
        radii vs. ln(REE/NORM) polynomical combination.

        Parameters
        ------------
        norm_to : :class:`str` | :class:`~pyrolite.geochem.norm.RefComp` | :class:`numpy.ndarray`
            Which reservoir to normalise REE data to (defaults to :code:`"Chondrite_PON"`).
        exclude : :class:`list`, :code:`["Pm", "Eu"]`
            Which REE elements to exclude from the fit. May wish to include Ce for minerals
            in which Ce anomalies are common.
        params : :class:`list`, :code:`None`
            Set of predetermined orthagonal polynomial parameters.
        degree : :class:`int`, 5
            Maximum degree polynomial fit component to include.
        append : :class:`list`, :code:`None`
            Whether to append lambda function (i.e. :code:`["function"]`).
        scale : :class:`str`
            Current units for the REE data, used to scale the reference dataset.

        Todo
        -----
            * Operate only on valid rows.
            * Add residuals, Eu, Ce anomalies as options to `append`.
            * Pre-build orthagonal parameters for REE combinations for calculation speed?

        References
        -----------
        .. [#ref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
               Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
               doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__


        See Also
        ---------
        :func:`~pyrolite.geochem.ind.get_ionic_radii`
        :func:`~pyrolite.util.math.lambdas`
        :func:`~pyrolite.util.math.OP_constants`
        :func:`~pyrolite.plot.REE_radii_plot`
        :func:`~pyrolite.geochem.norm.ReferenceCompositions`
        """
        obj = self._obj
        return lambda_lnREE(
            obj,
            norm_to=norm_to,
            exclude=exclude,
            params=params,
            degree=degree,
            append=append,
            scale=scale,
            **kwargs
        )

    def normalize_to(self, norm_to=None):
        """
        Normalise a dataframe to a given reference composition.

        Note
        ------
        This assumes that the two dataframes have equivalent units.
        """
        pass
        # if the normalisation requires modification to the reference composition
        # note it here in a logger.debug call
