"""
Submodule for working with geochemical data.
"""
import pandas as pd
import numpy as np

from ..util.meta import update_docstring_references
from ..util import units
from . import parse
from . import transform
from . import norm
from .ind import __common_elements__, __common_oxides__, REE, REY
from .ions import set_default_ionic_charges
from ..util.log import Handle

logger = Handle(__name__)

set_default_ionic_charges()

# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrochem")
@pd.api.extensions.register_dataframe_accessor("pyrochem")
class pyrochem(object):
    def __init__(self, obj):
        """Custom dataframe accessor for pyrolite geochemistry."""
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        pass

    # pyrolite.geochem.ind functions

    @property
    def list_elements(self):
        """
        Get the subset of columns which are element names.

        Returns
        --------
        :class:`list`

        Notes
        -------
        The list will have the same ordering as the source DataFrame.
        """
        fltr = self._obj.columns.isin(__common_elements__)
        return self._obj.columns[fltr].tolist()

    @property
    def list_isotope_ratios(self):
        """
        Get the subset of columns which are isotope ratios.

        Returns
        --------
        :class:`list`

        Notes
        -------
        The list will have the same ordering as the source DataFrame.
        """
        fltr = [parse.is_isotoperatio(c) for c in self._obj.columns]
        return self._obj.columns[fltr].tolist()

    @property
    def list_REE(self):
        """
        Get the subset of columns which are Rare Earth Element names.

        Returns
        --------
        :class:`list`

        Notes
        -------
        The returned list will reorder REE based on atomic number.
        """
        return [i for i in REE() if i in self._obj.columns]

    @property
    def list_REY(self):
        """
        Get the subset of columns which are Rare Earth Element names.

        Returns
        --------
        :class:`list`

        Notes
        -------
        The returned list will reorder REE based on atomic number.
        """
        return [i for i in REY() if i in self._obj.columns]

    @property
    def list_oxides(self):
        """
        Get the subset of columns which are oxide names.

        Returns
        --------
        :class:`list`

        Notes
        -------
        The list will have the same ordering as the source DataFrame.
        """
        fltr = self._obj.columns.isin(__common_oxides__)
        return self._obj.columns[fltr].tolist()

    @property
    def list_compositional(self):
        return list(self.list_oxides + self.list_elements)

    @property
    def elements(self):
        """
        Get an elemental subset of a DataFrame.

        Returns
        --------
        :class:`pandas.Dataframe`
        """
        return self._obj[self.list_elements]

    @elements.setter
    def elements(self, df):
        self._obj.loc[:, self.list_elements] = df

    @property
    def REE(self):
        """
        Get a Rare Earth Element subset of a DataFrame.

        Returns
        --------
        :class:`pandas.Dataframe`
        """
        return self._obj[self.list_REE]

    @REE.setter
    def REE(self, df):
        self._obj.loc[:, self.list_REE] = df

    @property
    def REY(self):
        """
        Get a Rare Earth Element + Yttrium subset of a DataFrame.

        Returns
        --------
        :class:`pandas.Dataframe`
        """
        return self._obj[self.list_REY]

    @REY.setter
    def REY(self, df):
        self._obj.loc[:, self.list_REY] = df

    @property
    def oxides(self):
        """
        Get an oxide subset of a DataFrame.

        Returns
        --------
        :class:`pandas.Dataframe`
        """
        return self._obj.loc[:, self.list_oxides]

    @oxides.setter
    def oxides(self, df):
        self._obj.loc[:, self.list_oxides] = df

    @property
    def isotope_ratios(self):
        """
        Get an isotope ratio subset of a DataFrame.

        Returns
        --------
        :class:`pandas.Dataframe`
        """
        return self._obj[self.list_isotope_ratios]

    @isotope_ratios.setter
    def isotope_ratios(self, df):
        self._obj.loc[:, self.list_isotope_ratios] = df

    @property
    def compositional(self):
        """
        Get an oxide & elemental subset of a DataFrame.

        Returns
        --------
        :class:`pandas.Dataframe`

        Notes
        ------
        This wil not include isotope ratios.
        """
        return self._obj.loc[:, self.list_compositional]

    @compositional.setter
    def compositional(self, df):
        self._obj.loc[:, self.list_compositional] = df

    # pyrolite.geochem.parse functions

    def parse_chem(self, abbrv=["ID", "IGSN"], split_on=r"[\s_]+"):
        """
        Convert column names to pyrolite-recognised elemental, oxide and isotope
        ratio column names where valid names are found.
        """
        self._obj.columns = parse.tochem(
            self._obj.columns, abbrv=abbrv, split_on=split_on
        )
        return self._obj

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
        """
        return parse.check_multiple_cation_inclusion(self._obj, exclude=exclude)

    # pyrolite.geochem.transform functions

    def to_molecular(self, renorm=True):
        """
        Converts mass quantities to molar quantities.

        Parameters
        -----------
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after converting to relative moles.

        Notes
        ------
        Does not convert units (i.e. mass% --> mol%; mass-ppm --> mol-ppm).

        Returns
        -------
        :class:`pandas.DataFrame`
            Transformed dataframe.
        """
        self._obj = transform.to_molecular(self._obj, renorm=renorm)
        return self._obj

    def to_weight(self, renorm=True):
        """
        Converts molar quantities to mass quantities.

        Parameters
        -----------
        renorm : :class:`bool`, :code:`True`
            Whether to renormalise the dataframe after converting to relative moles.

        Notes
        ------
        Does not convert units (i.e. mol% --> mass%; mol-ppm --> mass-ppm).

        Returns
        -------
        :class:`pandas.DataFrame`
            Transformed dataframe.
        """
        self._obj = transform.to_weight(self._obj, renorm=renorm)
        return self._obj

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
        self._obj = transform.devolatilise(self._obj, exclude=exclude, renorm=renorm)
        return self._obj

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
        return transform.elemental_sum(
            self._obj,
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
        return transform.aggregate_element(
            self._obj,
            to,
            total_suffix=total_suffix,
            logdata=logdata,
            renorm=renorm,
            molecular=molecular,
        )

    def recalculate_Fe(
        self, to="FeOT", renorm=False, total_suffix="T", logdata=False, molecular=False
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
        renorm : :class:`bool`, :code:`False`
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
        self._obj = transform.recalculate_Fe(
            self._obj,
            to,
            total_suffix=total_suffix,
            logdata=logdata,
            renorm=renorm,
            molecular=molecular,
        )
        return self._obj

    def get_ratio(self, ratio: str, alias: str = None, norm_to=None, molecular=False):
        """
        Add a ratio of components A and B, given in the form of string 'A/B'.
        Returned series be assigned an alias name.

        Parameters
        -----------
        ratio : :class:`str`
            String decription of ratio in the form A/B[_n].
        alias : :class:`str`
            Alternate name for ratio to be used as column name.
        norm_to : :class:`str` | :class:`pyrolite.geochem.norm.Composition`, `None`
            Reference composition to normalise to.
        molecular : :class:`bool`, :code:`False`
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with ratio appended.

        See Also
        --------
        :func:`~pyrolite.geochem.transform.add_MgNo`
        """
        return transform.get_ratio(
            self._obj, ratio, alias, norm_to=norm_to, molecular=molecular
        )

    def add_ratio(self, ratio: str, alias: str = None, norm_to=None, molecular=False):
        """
        Add a ratio of components A and B, given in the form of string 'A/B'.
        Returned series be assigned an alias name.

        Parameters
        -----------
        ratio : :class:`str`
            String decription of ratio in the form A/B[_n].
        alias : :class:`str`
            Alternate name for ratio to be used as column name.
        norm_to : :class:`str` | :class:`pyrolite.geochem.norm.Composition`, `None`
            Reference composition to normalise to.
        molecular : :class:`bool`, :code:`False`
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with ratio appended.

        See Also
        --------
        :func:`~pyrolite.geochem.transform.add_MgNo`
        """
        r = self.get_ratio(ratio, alias, norm_to=norm_to, molecular=molecular)
        self._obj[r.name] = r
        return self._obj

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
        transform.add_MgNo(
            self._obj,
            molecular=molecular,
            use_total_approx=use_total_approx,
            approx_Fe203_frac=approx_Fe203_frac,
            name=name,
        )
        return self._obj

    def lambda_lnREE(
        self,
        norm_to="ChondriteREE_ON",
        exclude=["Pm", "Eu"],
        params=None,
        degree=4,
        scale="ppm",
        sigmas=None,
        **kwargs
    ):
        r"""
        Calculates orthogonal polynomial coefficients (lambdas) for a given set of REE data,
        normalised to a specific composition [#localref_1]_. Lambda factors are given for the
        radii vs. ln(REE/NORM) polynomial combination.

        Parameters
        ----------
        norm_to : :class:`str` | :class:`~pyrolite.geochem.norm.Composition` | :class:`numpy.ndarray`
            Which reservoir to normalise REE data to (defaults to :code:`"ChondriteREE_ON"`).
        exclude : :class:`list`, :code:`["Pm", "Eu"]`
            Which REE elements to exclude from the *fit*. May wish to include Ce for minerals
            in which Ce anomalies are common.
        params : :class:`list` | :class:`str`, :code:`None`
            Pre-computed parameters for the orthogonal polynomials (a list of tuples).
            Optionally specified, otherwise defaults the parameterisation as in
            O'Neill (2016). If a string is supplied, :code:`"O'Neill (2016)"` or
            similar will give the original defaults, while :code:`"full"` will use all
            of the REE (including Eu) as a basis for the orthogonal polynomials.
        degree : :class:`int`, 4
            Maximum degree polynomial fit component to include.
        scale : :class:`str`
            Current units for the REE data, used to scale the reference dataset.
        sigmas : :class:`float` | :class:`numpy.ndarray` | :class:`pandas.Series`
            Value or 1D array of fractional REE uncertaintes (i.e.
            :math:`\sigma_{REE}/REE`).

        References
        ----------
        .. [#localref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
               Rare Earth Element Patterns in Basalts. J Petrology 57:1463–1508.
               doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__

        See Also
        --------
        :func:`~pyrolite.geochem.ind.get_ionic_radii`
        :func:`~pyrolite.util.lambdas.calc_lambdas`
        :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
        :func:`~pyrolite.plot.REE_radii_plot`
        """
        return transform.lambda_lnREE(
            self._obj,
            norm_to=norm_to,
            exclude=exclude,
            params=params,
            degree=degree,
            scale=scale,
            sigmas=sigmas,
            **kwargs,
        )

    def convert_chemistry(self, to=[], logdata=False, renorm=False, molecular=False):
        """
        Attempts to convert a dataframe with one set of components to another.

        Parameters
        ----------
        to : :class:`list`
            Set of columns to try to extract from the dataframe.

            Can also include a dictionary for iron speciation.
            See :func:`pyrolite.geochem.recalculate_Fe`.
        logdata : :class:`bool`, :code:`False`
            Whether chemical data has been log transformed. Necessary for aggregation
            functions.
        renorm : :class:`bool`, :code:`False`
            Whether to renormalise the data after transformation.
        molecular : :class:`bool`, :code:`False`
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with converted chemistry.

        Todo
        ----
            * Check for conflicts between oxides and elements
            * Aggregator for ratios
            * Implement generalised redox transformation.
            * Add check for dicitonary components (e.g. Fe) in tests
        """
        return transform.convert_chemistry(
            self._obj, to=to, logdata=logdata, renorm=renorm, molecular=molecular
        )  # can't update the source nicely here, need to assign output

    # pyrolite.geochem.norm functions

    def normalize_to(self, reference=None, units=None, convert_first=False):
        """
        Normalise a dataframe to a given reference composition.

        Parameters
        ----------
        reference : :class:`str` | :class:`~pyrolite.geochem.norm.Composition` | :class:`numpy.ndarray`
            Reference composition to normalise to.
        units : :class:`str`
            Units of the input dataframe, to convert the reference composition.
        convert_first : :class:`bool`
            Whether to first convert the referenece compostion before normalisation.
            This is useful where elements are presented as different components (e.g.
            Ti, TiO2).

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with normalised chemistry.

        Notes
        -----
        This assumes that dataframes have a single set of units.
        """

        if isinstance(reference, (str, norm.Composition)):
            if not isinstance(reference, norm.Composition):
                N = norm.get_reference_composition(reference)
            else:
                N = reference
            if units is not None:
                N.set_units(units)
            if convert_first:
                N.comp = transform.convert_chemistry(N.comp, self.list_compositional)
            norm_abund = N[self.list_compositional]
        else:  # list, iterable, pd.Index etc
            norm_abund = np.array(reference)
            assert len(norm_abund) == len(self.list_compositional)

        # this list should have the same ordering as the input dataframe
        return self._obj[self.list_compositional].div(norm_abund)

    def denormalize_from(self, reference=None, units=None):
        """
        De-normalise a dataframe from a given reference composition.

        Parameters
        ----------
        reference : :class:`str` | :class:`~pyrolite.geochem.norm.Composition` | :class:`numpy.ndarray`
            Reference composition which the composition is normalised to.
        units : :class:`str`
            Units of the input dataframe, to convert the reference composition.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with normalised chemistry.

        Notes
        -----
        This assumes that dataframes have a single set of units.
        """

        if isinstance(reference, (str, norm.Composition)):
            if not isinstance(reference, norm.Composition):
                N = norm.get_reference_composition(reference)
            else:
                N = reference
            if units is not None:
                N.set_units(units)
            N.comp = transform.convert_chemistry(N.comp, self.list_compositional)
            norm_abund = N[self.list_compositional]
        else:  # list, iterable, pd.Index etc
            norm_abund = np.array(reference)
            assert len(norm_abund) == len(self.list_compositional)

        return self._obj[self.list_compositional] * norm_abund

    def scale(self, in_unit, target_unit="ppm"):
        """
        Scale a dataframe from one set of units to another.

        Parameters
        ----------
        in_unit : :class:`str`
            Units to be converted from
        target_unit : :class:`str`, :code:`"ppm"`
            Units to scale to.

        Returns
        -------
        :class:`pandas.DataFrame`
            Dataframe with new scale.
        """
        return self._obj * units.scale(in_unit, target_unit)


pyrochem.lambda_lnREE = update_docstring_references(
    pyrochem.lambda_lnREE, ref="localref"
)
