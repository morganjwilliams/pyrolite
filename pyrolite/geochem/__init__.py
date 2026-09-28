"""
Submodule for working with geochemical data.
"""

from typing import overload

import numpy as np
import pandas as pd
import periodictable as pt

from pyrolite.geochem.norm import Composition

from ..util import units
from ..util.log import Handle
from ..util.meta import update_docstring_references
from . import norm, parse, transform
from .ind import REE, REY, _common_elements, _common_oxides
from .ions import set_default_ionic_charges

logger = Handle(__name__)

set_default_ionic_charges()


# note that only some of these methods will be valid for series
@pd.api.extensions.register_series_accessor("pyrochem")
@pd.api.extensions.register_dataframe_accessor("pyrochem")
class pyrochem:
    def __init__(self, obj: pd.Series | pd.DataFrame):
        """Custom dataframe accessor for pyrolite geochemistry."""
        self._validate(obj)
        self._obj = obj

    @property
    def _selection_index(self) -> pd.Index:
        """
        Get the seleciton index of the current object.

        Best used as a property, generated on demand,
        otherwise will have memory from initial
        object instantiation.
        """
        return (
            self._obj.columns
            if isinstance(self._obj, pd.DataFrame)
            else self._obj.index
        )

    @staticmethod
    def _validate(obj):
        pass

    # pyrolite.geochem.ind functions

    @property
    def list_elements(self) -> list[str]:
        """
        Get the subset of columns which are element names.

        Returns
        --------
        list

        Notes
        -------
        The list will have the same ordering as the source DataFrame.
        """
        return [i for i in self._selection_index if i in _common_elements]

    @property
    def list_isotope_ratios(self) -> list[str]:
        """
        Get the subset of columns which are isotope ratios.

        Returns
        --------
        list

        Notes
        -------
        The list will have the same ordering as the source DataFrame.
        """
        return [c for c in self._selection_index if parse.is_isotoperatio(c)]

    @property
    def list_REE(self) -> list[str]:
        """
        Get the subset of columns which are Rare Earth Element names.

        Returns
        --------
        list

        Notes
        -------
        The returned list will reorder REE based on atomic number.
        """
        return [
            i
            for i in REE(dropPm=("Pm" not in self._selection_index))
            if i in self._selection_index
        ]

    @property
    def list_REY(self) -> list[str]:
        """
        Get the subset of columns which are Rare Earth Element names.

        Returns
        --------
        list

        Notes
        -------
        The returned list will reorder REE based on atomic number.
        """
        return [
            i
            for i in REY(dropPm=("Pm" not in self._selection_index))
            if i in self._selection_index
        ]

    @property
    def list_oxides(self) -> list[str]:
        """
        Get the subset of columns which are oxide names.

        Returns
        --------
        list

        Notes
        -------
        The list will have the same ordering as the source DataFrame/Series.
        """
        return [i for i in self._selection_index if i in _common_oxides]

    @property
    def list_compositional(self) -> list[str]:
        return list(self.list_oxides + self.list_elements)

    @overload
    @property
    def elements(self) -> pd.Series: ...
    @property
    def elements(self) -> pd.DataFrame:
        """
        Get an elemental subset of a DataFrame.

        Returns
        --------
        pandas.DataFrame | pandas.Series
        """
        return self._obj[self.list_elements]

    @elements.setter
    def elements(self, df: pd.DataFrame | pd.Series | np.ndarray):
        self._obj[self.list_elements] = df

    @overload
    @property
    def REE(self) -> pd.Series: ...
    @property
    def REE(self) -> pd.DataFrame:
        """
        Get a Rare Earth Element subset of a DataFrame.

        Returns
        --------
        pandas.DataFrame | pd.Series
        """
        return self._obj[self.list_REE]

    @REE.setter
    def REE(self, df: pd.DataFrame | pd.Series | np.ndarray):
        self._obj[self.list_REE] = df

    @overload
    @property
    def REY(self) -> pd.Series: ...
    @property
    def REY(self) -> pd.DataFrame:
        """
        Get a Rare Earth Element + Yttrium subset of a DataFrame.

        Returns
        --------
        pandas.DataFrame | pandas.Series
        """
        return self._obj[self.list_REY]

    @REY.setter
    def REY(self, df: pd.DataFrame | pd.Series | np.ndarray):
        self._obj[self.list_REY] = df

    @overload
    @property
    def oxides(self) -> pd.Series: ...
    @property
    def oxides(self) -> pd.DataFrame:
        """
        Get an oxide subset of a DataFrame.

        Returns
        --------
        pandas.DataFrame | pandas.Series
        """
        return self._obj[self.list_oxides]

    @oxides.setter
    def oxides(self, df: pd.DataFrame | pd.Series | np.ndarray):
        self._obj[self.list_oxides] = df

    @overload
    @property
    def isotope_ratios(self) -> pd.Series: ...
    @property
    def isotope_ratios(self) -> pd.DataFrame:
        """
        Get an isotope ratio subset of a DataFrame.

        Returns
        --------
        pandas.DataFrame | pandas.Series
        """
        return self._obj[self.list_isotope_ratios]

    @isotope_ratios.setter
    def isotope_ratios(self, df: pd.DataFrame | pd.Series | np.ndarray):
        self._obj[self.list_isotope_ratios] = df

    @overload
    @property
    def compositional(self) -> pd.Series: ...
    @property
    def compositional(self) -> pd.DataFrame:
        """
        Get an oxide & elemental subset of a DataFrame.

        Returns
        --------
        pandas.DataFrame | pandas.Series

        Notes
        ------
        This wil not include isotope ratios.
        """
        return self._obj[self.list_compositional]

    @compositional.setter
    def compositional(self, df: pd.DataFrame | pd.Series | np.ndarray):
        self._obj[self.list_compositional] = df

    # pyrolite.geochem.parse functions

    def parse_chem(
        self, abbrv: list[str] | None = None, split_on: str = r"[\s_]+"
    ) -> pd.DataFrame | pd.Series:
        """
        Convert column names to pyrolite-recognised elemental, oxide and isotope
        ratio column names where valid names are found.
        """
        if abbrv is None:
            abbrv = ["ID", "IGSN"]
        self._obj.columns = parse.tochem(
            self._obj.columns, abbrv=abbrv, split_on=split_on
        )
        return self._obj

    def check_multiple_cation_inclusion(self, exclude: list[str] | None = None) -> set:
        """
        Returns cations which are present in both oxide and elemental form.

        Parameters
        -----------
        exclude : list
            List of components to exclude from the duplication check.

        Returns
        --------
        set
            Set of elements for which multiple components exist in the dataframe.
        """
        if exclude is None:
            exclude = ["LOI", "FeOT", "Fe2O3T"]
        return parse.check_multiple_cation_inclusion(self._obj, exclude=exclude)

    # pyrolite.geochem.transform functions

    def to_molecular(self, renorm: bool = True) -> pd.DataFrame | pd.Series:
        """
        Converts mass quantities to molar quantities.

        Parameters
        -----------
        renorm : bool
            Whether to renormalise the dataframe after converting to relative moles.

        Notes
        ------
        Does not convert units (i.e. mass% --> mol%; mass-ppm --> mol-ppm).

        Returns
        -------
        pandas.DataFrame | pandas.Series
            Transformed dataframe.
        """
        return transform.to_molecular(self._obj, renorm=renorm)

    def to_weight(self, renorm: bool = True) -> pd.DataFrame | pd.Series:
        """
        Converts molar quantities to mass quantities.

        Parameters
        -----------
        renorm : bool
            Whether to renormalise the dataframe after converting to relative moles.

        Notes
        ------
        Does not convert units (i.e. mol% --> mass%; mol-ppm --> mass-ppm).

        Returns
        -------
        pandas.DataFrame | pandas.Series
            Transformed dataframe.
        """
        return transform.to_weight(self._obj, renorm=renorm)

    def devolatilise(
        self, exclude: list[str] | None = None, renorm: bool = True
    ) -> pd.DataFrame | pd.Series:
        """
        Recalculates components after exclusion of volatile phases (e.g. H2O, CO2).

        Parameters
        -----------
        exclude : list
            Components to exclude from the dataset.
        renorm : bool
            Whether to renormalise the dataframe after devolatilisation.

        Returns
        -------
        pandas.DataFrame | pandas.Series
            Transformed dataframe.
        """
        if exclude is None:
            exclude = ["H2O", "H2O_PLUS", "H2O_MINUS", "CO2", "LOI"]
        return transform.devolatilise(self._obj, exclude=exclude, renorm=renorm)

    def elemental_sum(
        self,
        component: str | None = None,
        to: str | None = None,
        total_suffix: str = "T",
        logdata: bool = False,
        molecular: bool = False,
    ) -> pd.Series | float:
        """
        Sums abundance for a cation to a single series, starting from a
        dataframe containing multiple componnents with a single set of units.

        Parameters
        ----------
        component : str
            Component indicating which element to aggregate.
        to : str
            Component to cast the output as.
        logdata : bool
            Whether data has been log transformed.
        molecular : bool
            Whether to perform a sum of molecular data.

        Returns
        -------
        pandas.Series | float
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
        self,
        to: str | pt.core.Element | pt.formulas.Formula | dict,
        total_suffix: str = "T",
        logdata: bool = False,
        renorm: bool = False,
        molecular: bool = False,
    ) -> pd.Series | float:
        """
        Aggregates cation information from oxide and elemental components to either a
        single species or a designated mixture of species.

        Parameters
        ----------
        to : str | periodictable.core.Element | periodictable.formulas.Formula | dict
            Component(s) to convert to. If one component is specified, the element will be
            converted to the target species.

            If more than one component is specified with proportions in a dictionary
            (e.g. `{'FeO': 0.9, 'Fe2O3': 0.1}`), the components will be split as a
            fraction of the elemental sum.
        renorm : bool
            Whether to renormalise the dataframe after recalculation.
        total_suffix : str
            Suffix of 'total' variables. E.g. 'T' for FeOT, Fe2O3T.
        logdata : bool
            Whether the data has been log transformed.
        molecular : bool
            Whether to perform a sum of molecular data.

        Notes
        -------
        This won't convert units, so need to start from single set of units.

        Returns
        -------
        pandas.Series
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
        self,
        tol: str | pt.core.Element | pt.formulas.Formula | dict = "FeOT",
        renorm: bool = False,
        total_suffix: str = "T",
        logdata: bool = False,
        molecular: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        Recalculates abundances of iron, and normalises a dataframe to contain  either
        a single species, or multiple species in certain proportions.

        Parameters
        -----------
        to : str | periodictable.core.Element | periodictable.formulas.Formula | dict
            Component(s) to convert to.

            If one component is specified, all iron will be
            converted to the target species.

            If more than one component is specified with proportions in a dictionary
            (e.g. `{'FeO': 0.9, 'Fe2O3': 0.1}`), the components will be split as a
            fraction of Fe.
        renorm : bool
            Whether to renormalise the dataframe after recalculation.
        total_suffix : str
            Suffix of 'total' variables. E.g. 'T' for FeOT, Fe2O3T.
        logdata : bool
            Whether the data has been log transformed.
        molecular : bool
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        pandas.DataFrame
            Transformed dataframe.
        """
        return transform.recalculate_Fe(
            self._obj,
            to,
            total_suffix=total_suffix,
            logdata=logdata,
            renorm=renorm,
            molecular=molecular,
        )

    def get_ratio(
        self,
        ratio: str,
        alias: str | None = None,
        norm_to: str | Composition | None = None,
        molecular: bool = False,
    ) -> pd.Series | float:
        """
        Add a ratio of components A and B, given in the form of string 'A/B'.
        Returned series be assigned an alias name.

        Parameters
        -----------
        ratio : str
            String decription of ratio in the form A/B[_n].
        alias : str
            Alternate name for ratio to be used as column name.
        norm_to : str | pyrolite.geochem.norm.Composition
            Reference composition to normalise to.
        molecular : bool
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        pandas.Series | float
            Ratio series.

        See Also
        --------
        :func:`~pyrolite.geochem.transform.add_MgNo`
        """
        return transform.get_ratio(
            self._obj, ratio, alias, norm_to=norm_to, molecular=molecular
        )

    def add_ratio(
        self, ratio: str, alias: str | None = None, norm_to=None, molecular=False
    ) -> pd.DataFrame | pd.Series:
        """
        Add a ratio of components A and B, given in the form of string 'A/B'.
        Returned series be assigned an alias name.

        Parameters
        -----------
        ratio : str
            String decription of ratio in the form A/B[_n].
        alias : str
            Alternate name for ratio to be used as column name.
        norm_to : str | pyrolite.geochem.norm.Composition
            Reference composition to normalise to.
        molecular : bool
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        pandas.DataFrame
            Dataframe with ratio appended.

        See Also
        --------
        :func:`~pyrolite.geochem.transform.add_MgNo`
        """
        self._obj[ratio] = self.get_ratio(
            ratio, alias, norm_to=norm_to, molecular=molecular
        )
        return self._obj

    def add_MgNo(
        self,
        molecular: bool = False,
        use_total_approx: bool = False,
        approx_Fe203_frac: float = 0.1,
        name: str = "Mg#",
    ) -> pd.DataFrame | pd.Series:
        """
        Append the magnesium number to a dataframe.

        Parameters
        ----------
        molecular : bool, `False`
            Whether the input data is molecular.
        use_total_approx : bool, `False`
            Whether to use an approximate calculation using total iron rather than just FeO.
        approx_Fe203_frac : float
            Fraction of iron which is oxidised, used in approximation mentioned above.
        name : str
            Name to use for the Mg Number column.

        Returns
        -------
        pandas.DataFrame
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
        norm_to: str | Composition | None = "ChondriteREE_ON",
        excludel: list[str] | None = None,
        params: list | str | None = None,
        degree: int = 4,
        scalel: str = "ppm",
        sigmas: np.ndarray[tuple[int], np.dtype[np.floating]] = None,
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        r"""
        Calculates orthogonal polynomial coefficients (lambdas) for a given set of REE data,
        normalised to a specific composition [#localref_1]_. Lambda factors are given for the
        radii vs. ln(REE/NORM) polynomial combination.

        Parameters
        ----------
        norm_to : str | pyrolite.geochem.norm.Composition | numpy.ndarray
            Which reservoir to normalise REE data to (defaults to `"ChondriteREE_ON"`).
        exclude : list
            Which REE elements to exclude from the *fit*. May wish to include Ce for minerals
            in which Ce anomalies are common.
        params : list | str
            Pre-computed parameters for the orthogonal polynomials (a list of tuples).
            Optionally specified, otherwise defaults the parameterisation as in
            O'Neill (2016). If a string is supplied, `"O'Neill (2016)"` or
            similar will give the original defaults, while `"full"` will use all
            of the REE (including Eu) as a basis for the orthogonal polynomials.
        degree : int, 4
            Maximum degree polynomial fit component to include.
        scale : str
            Current units for the REE data, used to scale the reference dataset.
        sigmas : float | numpy.ndarray | pandas.Series
            Value or 1D array of fractional REE uncertaintes (i.e.
            :math:`\sigma_{REE}/REE`).

        References
        ----------
        .. [#localref_1] O’Neill HSC (2016) The Smoothness and Shapes of Chondrite-normalized
               Rare Earth Element Patterns in Basalts. J Petrology 57:1463-1508.
               doi: `10.1093/petrology/egw047 <https://dx.doi.org/10.1093/petrology/egw047>`__

        See Also
        --------
        :func:`~pyrolite.geochem.ind.get_ionic_radii`
        :func:~pyrolite.util.lambdas.calc_lambdas`
        :func:`~pyrolite.util.lambdas.orthogonal_polynomial_constants`
        :func:`~pyrolite.plot.REE_radii_plot`
        """
        if exclude is None:
            exclude = ["Pm", "Eu"]
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

    def convert_chemistry(
        self,
        to: list[str | dict] | None = None,
        logdata: bool = False,
        renorm: bool = False,
        molecular: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        Attempts to convert a dataframe with one set of components to another.

        Parameters
        ----------
        to : list
            Set of columns to try to extract from the dataframe.

            Can also include a dictionary for iron speciation.
            See :func:`pyrolite.geochem.recalculate_Fe`.
        logdata : bool
            Whether chemical data has been log transformed. Necessary for aggregation
            functions.
        renorm : bool
            Whether to renormalise the data after transformation.
        molecular : bool
            Flag that data is in molecular units, rather than weight units.

        Returns
        -------
        pandas.DataFrame
            Dataframe with converted chemistry.

        Todo
        ----
        * Check for conflicts between oxides and elements
        * Aggregator for ratios
        * Implement generalised redox transformation.
        * Add check for dicitonary components (e.g. Fe) in tests
        """
        if to is None:
            to = []
        return transform.convert_chemistry(
            self._obj,
            to=to,
            logdata=logdata,
            renorm=renorm,
            molecular=molecular,
        )  # can't update the source nicely here, need to assign output

    # pyrolite.geochem.norm functions

    def normalize_to(
        self,
        reference: str
        | Composition
        | np.ndarray[tuple[int], np.dtype[np.floating]]
        | None = None,
        units: str | None = None,
        convert_first: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        Normalise a dataframe to a given reference composition.

        Parameters
        ----------
        reference : str | pyrolite.geochem.norm.Composition | numpy.ndarray
            Reference composition to normalise to.
        units : str
            Units of the input dataframe, to convert the reference composition.
        convert_first : bool
            Whether to first convert the referenece compostion before normalisation.
            This is useful where elements are presented as different components (e.g.
            Ti, TiO2).

        Returns
        -------
        pandas.DataFrame
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

    def denormalize_from(
        self,
        reference: str | Composition | np.ndarray[tuple[int], np.dtype[np.floating]],
        units: str | None = None,
    ) -> pd.DataFrame | pd.Series:
        """
        De-normalise a dataframe from a given reference composition.

        Parameters
        ----------
        reference : str | `~pyrolite.geochem.norm.Composition` | numpy.ndarray
            Reference composition which the composition is normalised to.
        units : str
            Units of the input dataframe, to convert the reference composition.

        Returns
        -------
        pandas.DataFrame
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
            # N.comp = transform.convert_chemistry(
            #     N.comp.to_frame().T, self.list_compositional
            # ).iloc[0]
            N.comp = transform.convert_chemistry(N.comp, self.list_compositional)
            norm_abund = N[self.list_compositional]
        else:  # list, iterable, pd.Index etc
            norm_abund = np.array(reference)
            assert len(norm_abund) == len(self.list_compositional)

        return self._obj[self.list_compositional] * norm_abund

    def scale(self, in_unit: str, target_unit: str = "ppm") -> pd.DataFrame | pd.Series:
        """
        Scale a dataframe from one set of units to another.

        Parameters
        ----------
        in_unit : str
            Units to be converted from
        target_unit : str, `"ppm"`
            Units to scale to.

        Returns
        -------
        pandas.DataFrame
            Dataframe with new scale.
        """
        return self._obj * units.scale(in_unit, target_unit)


pyrochem.lambda_lnREE = update_docstring_references(
    pyrochem.lambda_lnREE, ref="localref"
)
