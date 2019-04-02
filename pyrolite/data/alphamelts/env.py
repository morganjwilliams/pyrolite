from pyrolite.util.text import normalise_whitespace, to_width
from pathlib import Path


def mvar(**kwargs):
    """A data dictonary specifying MELTS environment variables."""
    d = dict(
        validate=None,
        desc=None,
        type=None,
        units=None,
        set=True,
        default=None,
        overridden_by=None,
        dependent_on=None,
        priorty=0,
    )
    d.update(kwargs)
    return d


def description(text, width=69):
    return to_width(normalise_whitespace(text), width=width)


def filepath_validator(path):
    """Validator for file paths."""
    if isinstance(path, str):
        path = Path(path)
    return path.exists() and path.isfile()


def dirpath_validator(path):
    """Validator for directory paths."""
    if isinstance(path, str):
        path = Path(path)
    return path.exists() and path.isdir()


def numeric_validator(value):
    """Validator for numeric values."""
    return isinstance(float(value), float) or isinstance(int(value), int)


def typecast(type, item):
    """Typecasts argument input for lambda functions."""

    if isinstance(item, list):
        # Return each of the function items as Typecast
        # equivalents
        return [lambda x: f(type(x)) if callable(f) else f for f in item]
    else:
        return lambda x: item(type(x))


## The following is directly form the alphaMELTS Documentation v1.8
## Written by Paula Antoshechkina and Paul Asimow October 13, 2017
MELTS_environment_variables = {  # Solution Models
    "VERSION": mvar(
        default="pMELTS",
        validate=lambda x: x in ["MELTS", "pMELTS"],
        desc=description(
            """Set to choose the liquid thermodynamic model.
           pMELTS is only recommended for peridotite bulk
           compositions between 1 and 4 GPa. If doing pHMELTS
           calculations then use ‘pMELTS’ for melting and
           ‘MELTS’ for low-pressure crystallization."""
        ),
    ),
    "OLD_GARNET": mvar(
        set=False,
        type=bool,
        desc=description(
            """Uses the old, incorrect, garnet model. When set,
              the behavior will resemble that of all GUI
              versions of MELTS released before 2005 (including
              Java MELTS) and the default behavior of
              Adiabat_1ph versions 1.4 – 2.X. When unset,
              will behave as if ADIABAT_FIX_GARNET was set in
              older (pre-3.0) versions of Adiabat_1ph."""
        ),
    ),
    "OLD_SPINEL": mvar(
        set=False,
        type=bool,
        desc=description(
            """Use the spinel model from Sack and
              Ghiorso [1991], with perfectly ideal
              mixing of volumes. When set, gives the
              same behavior as Java MELTS and
              earlier (pre-3.0) versions of
              Adiabat_1ph. When unset, uses the
              spinel volume model as formulated in
              Ghiorso and Sack [1991]."""
        ),
    ),
    "OLD_BIOTITE": mvar(
        set=False,
        type=bool,
        desc=description(
            """Uses the biotite model from Sack and Ghiorso
           [1989] instead of the default alphaMELTS one.
           When set, the program uses the same code as
           GUI versions of MELTS (2005-present). The
           behavior may differ slightly from earlier GUI
           versions of MELTS as a small error was fixed."""
        ),
    ),
    "2_AMPH": mvar(
        set=False,
        type=bool,
        desc=description(
            """Uses separate orthoamphibole and clinoamphibole
              phases. When set, gives the same behavior as Java
              MELTS and Corba MELTS applets, and earlier versions
              of Adiabat_1ph (which inherited its code from the
              Java MELTS branch). When unset, will use a single
              cummingtonite-grunerite-tremolite solution and deduce
              the amphibole structure from the energetics."""
        ),
    ),
    "NO_CHLORITE": mvar(
        set=False,
        type=bool,
        desc=description(
            """As of version 1.6, the chlorite model of
               Hunziker [2003] is included in alphaMELTS
               (see Smith et al. [2007]). To reproduce previous
               results for bulk compositions that may stabilize
               chlorite, add a ‘Suppress: chlorite’ line to the
               melts_file or set this variable."""
        ),
    ),
    # P-T Path and fO2 control
    "MODE": mvar(
        validate=lambda x: x
        in [
            "geothermal",
            "isenthalpic",
            "isentropic",
            "isobaric",
            "isochoric",
            "isothermal",
            "ptfile",
            "ptgrid",
            "tpgrid",
        ],
        default="isentropic",
        desc=description(
            """Sets the calculation mode. This variable is case
             insensitive. ‘PTfile’ (‘PTPath’ is also accepted
             for backwards compatibility) reads in P and T from
             a file; see below. ‘PTgrid’ and ‘TPgrid’ perform
             calculations on a grid bounded by the starting T
             and ALPHAMELTS_MINT/MAXT, and by the starting P and
             ALPHAMELTS_MINP/MAXP. For ‘PTgrid’ the T-loop is
             inside the P-loop, whereas ‘TPgrid’ is the other
             way round; in either case the (superliquidus or
             subsolidus) starting solution migrates with the
             outer loop. Generally, unless liquid is suppressed,
             ‘PTgrid’ with ALPHAMELTS_DELTAT < 0 will be the
             easiest combination. For other modes, there are
             various ways to initialize P, T and / or reference
             entropy, enthalpy or volume, and the thermodynamic
             path is set using ALPHAMELTS_DELTAP and / or
             ALPHAMELTS_DELTAT."""
        ),
    ),
    "PTPATH_FILE": mvar(
        set=False,
        validate=filepath_validator,
        desc=description(
            """Gives the name of the ptpath_file, which is a simple
            space delimited text file with one ‘P_value T_value’
            pair per line. If ALPHAMELTS_DELTAP and
            ALPHAMELTS_DELTAT are both zero the user will be asked
            for a maximum number of iterations to perform."""
        ),
    ),
    "DELTAP": mvar(
        validate=numeric_validator,
        type=float,
        default=1000.0,
        units="bars",
        desc=description(
            """This sets the pressure increment for isentropic,
               isothermal, geothermal or phase diagram mode.
               This is a signed number; i.e., a positive value steps
               upwards in P, negative steps down. If using a ptpath_file,
               a non-zero ALPHAMELTS_DELTAP means the whole will be read
               in and executed. Setting ALPHAMELTS_DELTAP to zero, with a
               non-zero value for ALPHAMELTS_DELTAT, means that phase
               diagram mode will step in temperature instead of pressure
               (except for liquid)."""
        ),
    ),
    "DELTAT": mvar(
        validate=numeric_validator,
        default=10.0,
        type=float,
        units="Degrees Celsius",
        desc=description(
            """This sets the temperature increment for isobaric,
               isochoric, geothermal or phase diagram mode. This is a
               signed number; i.e., a positive value steps upwards in T,
               negative steps down. If using a ptpath_file, a non-zero
               ALPHAMELTS_DELTAT means the whole will be read in and
               executed. For phase diagram mode to step in temperature
               ALPHAMELTS_DELTAP must be set to zero."""
        ),
    ),
    "MAXP": mvar(
        validate=numeric_validator,
        type=float,
        default=lambda env: [30000, 40000][env["VERSION"] != "MELTS"],
        dependent_on=["VERSION"],
        units="bars",
        desc=description(
            """Sets the maximum pressure the program will go to on
                 execution."""
        ),
    ),
    "MINP": mvar(
        validate=numeric_validator,
        type=float,
        default=lambda env: [1, 10000][env["VERSION"] != "MELTS"],
        dependent_on=["VERSION"],
        units="bars",
        desc=description(
            """Sets the minimum pressure the program will go to on
                 execution."""
        ),
    ),
    "MAXT": mvar(
        validate=numeric_validator,
        type=float,
        default=2000,
        units="Degrees Celsius",
        desc=description(
            """Sets the maximum temperature the program will go to on
                  execution."""
        ),
    ),
    "MINT": mvar(
        validate=numeric_validator,
        type=float,
        default=0,
        units="Degrees Celsius",
        desc=description(
            """Sets the minimum temperature the program will go to on
                  execution."""
        ),
    ),
    "ALTERNATIVE_FO2": mvar(
        set=False,
        type=bool,
        desc=description(
            """Normally, the parameterization of Kress and
            Carmichael [1991] is used to calculate fO2 in the
            liquid. If conditions are subsolidus or liquid is
            suppressed then the approach detailed in Asimow and
            Ghiorso [1998] is used to construct an appropriate
            redox reaction to solve for fO2 of the bulk
            assemblage. If this environment variable is set,
            however, then the method of Asimow and Ghiorso
            [1998] is used to calculate fO2 regardless of
            whether liquid is present and so, in theory, allows
            for a smoother transition across the solidus. """
        ),
    ),
    "LIQUID_FO2": mvar(
        set=False,
        type=bool,
        overridden_by=["LIQUID_FO2"],
        desc=description(
            """The method of Asimow and Ghiorso
           [1998] is computationally more involved than the
           parameterization of Kress and Carmichael [1991] and it
           is not uncommon for fO2 calculations to be successful
           with liquid present but fail subsolidus. It is possible
           to turn off the fO2 buffer manually (option 5).
           Alternatively, if this environment variable is set then
           the fO2 buffer, as formulated in Kress and Carmichael
           [1991], will only be imposed when liquid is present.
           Note that setting this variable does not change the
           fO2 buffer setting (e.g. ‘FMQ’); the program just ignores
            the flag if no liquid is around."""
        ),
    ),
    "IMPOSE_FO2": mvar(
        set=False,
        type=bool,
        desc=description(
            """Normally, for isentropic, isenthalpic and isochoric
           modes any fO2 buffer will be switched off on execution
           once the reference entropy, enthalpy or volume has been
           set (usually after the first calculation or before if set
           manually). If this environment variable is set then the
           program will alternate between (1) an unbuffered
           isenthalpic / isentropic / isochoric step and (2) an
           isobaric / isothermal fO2 buffered step. Overall this
           approximates an isenthalpic, isentropic or isochoric
           path with a desired fO2 buffer."""
        ),
    ),
    "FO2_PRESSURE_TERM": mvar(
        set=False,
        type=bool,
        desc=description(
            """In most versions of MELTS, including Corba
              MELTS and alphaMELTS, by default, reference fO2
              buffers in the system Fe-Si-O are calculated from
              the equations given in Myers and Eugster [1983].
              In the standalone GUI version of MELTS a pressure
              term is added to the fayalite-magnetite-quartz
              buffer (FMQ) to give:

              log(fO2) = − 24441.9/T + 0.110 (P − 1)/T + 8.290

              When this environment variable is set then
              alphaMELTS uses the equation as shown whereas by
              default the second term is omitted. Inclusion of
              the pressure term can have subtle effects on the
              stability of certain phases such as pyroxenes.
              Note that in Kessel et al. [2001] the coefficient
              on the pressure term is 0.05, rather than 0.110;
              the value of 0.110 was chosen by Ghiorso to
              maintain consistency with Berman [1988]."""
        ),
    ),
    # Open vs closed system
    "CONTINUOUS_MELTING": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default, batch melting equations are used.
               Setting this environment variable will change the
               melting mode to continuous or fractional, where
               melt is extracted after each equilibrium. Set
               ALPHAMELTS_MINF to 0 for perfect fractional
               melting. In practice though, the program will
               run more smoothly if ALPHAMELTS_MINF is slightly
               greater than 0. See the next four environment
               variables’ entries for more details."""
        ),
    ),
    "MINF": mvar(
        default=0.005,
        type=float,
        validator=typecast(
            float, [numeric_validator, lambda x: (x >= 0.0) & (x < 1.0)]
        ),
        overridden_by=["MINPHI"],
        desc=description(
            """If ALPHAMELTS_CONTINUOUS_MELTING is set, then by default a
         fixed melt fraction, by mass, marks the threshold above which
         melt is extracted. This variable is used to change the amount
         of melt retained. If the current melt fraction is less than
         ALPHAMELTS_MINF then all the melt will be retained until the
         next step, otherwise the amount of melt removed (approximately
         F - ALPHAMELTS_MINF) will be adjusted so such that the melt
         fraction is exactly ALPHAMELTS_MINF after extraction."""
        ),
    ),
    "MINPHI": mvar(
        set=False,
        type=float,
        validator=typecast(
            float, [numeric_validator, lambda x: (x >= 0.0) & (x < 1.0)]
        ),
        overridden_by=["CONTINUOUS_RATIO"],
        desc=description(
            """If ALPHAMELTS_CONTINUOUS_MELTING is set, then set this
           environment variable controls the retained melt fraction,
           by volume i.e. the ‘residual porosity’. If the current melt
           fraction is less than ALPHAMELTS_MINPHI then all the melt
           will be retained until the next step, otherwise the amount
           of melt removed (approximately phi - ALPHAMELTS_MINPHI) will
           be adjusted so that the melt fraction, by volume, is exactly
           ALPHAMELTS_MINPHI after extraction."""
        ),
    ),
    "CONTINUOUS_RATIO": mvar(
        set=False,
        type=float,
        validator=typecast(
            float, [numeric_validator, lambda x: (x >= 0.0) & (x < 1.0)]
        ),
        overridden_by=["CONTINUOUS_VOLUME"],
        desc=description(
            """This implements another alternative definition
             of continuous melting. Instead of extracting all
             liquid above a fixed mass or volume fraction, this
             option, if set, causes the program to multiply the
             liquid mass by a fixed ratio."""
        ),
    ),
    "CONTINUOUS_VOLUME": mvar(
        set=False,
        type=float,
        desc=description(
            """"This option, if set, extracts the required
              amount of melt to retain a constant total volume.
              This reference volume is set the first time
              melting occurs and is equal to the solid volume
              plus whatever melt volume is retained according
              to the ALPHAMELTS_MINPHI variable (or a default
              value of 0.002 if that is unset). Only for
              isobaric or isothermal calculations. Note that
              this is not an ‘isochoric’ calculation as far
              as alphaMELTS is concerned because melting is
              still allowed to cause expansion; this option
              only controls how much melt must be extracted
              to return to the original volume and, if
              necessary, also adjusts pressure (for isothermal
              calculations) or temperature (for isobaric
              calculations) to maintain equilibrium."""
        ),
    ),
    "FRACTIONATE_SOLIDS": mvar(
        set=False,
        type=bool,
        desc=description(
            """To turn on fractional crystallization of all
               solid phases, set this option to true (does not
               include water, see below). Do not use this
               option if you wish to selectively fractionate
               just a few phases; instead put ‘Fractionate:
               phase’ lines in you melts_file (see the ‘MELTS
               file’ section) or adjust individual phase
               settings with menu option 8."""
        ),
    ),
    "MASSIN": mvar(
        default=0.001,
        type=float,
        validator=typecast(
            float, [numeric_validator, lambda x: (x >= 0.0) & (x < 1.0)]
        ),
        overridden_by=["MINW"],
        desc=description(
            """Set to the mass in grams of each solid phase that is
                   retained during fractional crystallization. An increased
                   value may help to stabilize the calculation. A smaller value
                   can also be used, but a minimum of 10-6 grams is recommended.
                   nce the phase is no longer in the equilibrium assemblage it
                   will be completely exhausted, regardless of the
                   ALPHAMELTS_MASSIN value."""
        ),
    ),
    "FRACTIONATE_WATER": mvar(
        set=False,
        type=bool,
        desc=description(
            """To remove free water at each calculation stage
                              in an analogous way to how melt is removed during
                              continuous melting, set this variable true. (Note
                              that, in MELTS and pMELTS, water is treated like
                              a ‘solid’, in the sense that it is not melt, so
                              you can achieve the same effect by putting a
                              ‘Fractionate: water’ line in your melts_file.
                              However, water is treated differently from the
                              other mineral phases in that it may be extracted
                              during melting or crystallization.)"""
        ),
    ),
    "MINW": mvar(
        set=False,
        type=float,
        validator=typecast(
            float, [numeric_validator, lambda x: (x >= 0.0) & (x < 1.0)]
        ),
        desc=description(
            """Set to the proportion of retained water, relative to the
                 total system mass. Works in a similar way, for water, as
                 ALPHAMELTS_MINF does for melt; may be set to exactly zero. If
                 not set then fractionation of water is treated in a similar way
                 to fractionation of a solid phase i.e. a nominal mass of 10-4
                 grams is retained at each stage to stabilize the calculation;
                 the smaller mass of water retained, compared to other solid
                 phases, reflects its significantly lower molecular mass."""
        ),
    ),
    "FRACTIONATE_TARGET": mvar(
        set=False,
        type=float,
        desc=description(
            """During normal forward fractional
                               crystallization a (negative) increment is added
                               to the temperature at each step,
                               ALPHAMELTS_DELTAT, and the run is terminated when
                               the temperature goes below ALPHAMELTS_MINT. If
                               this environment variable is set then forward or
                               backward fractionation is performed so that the
                               MgO content (in wt %) or the Mg# (in mol %) hits
                               a particular target; see the next two entries.
                               When option 3 is called alphaMELTS will perform
                               a single calculation for the current P-T
                               conditions, in the normal way. When option 4 is
                               used it will first try to find the liquidus
                               instead and use that to decide whether forward
                               or backwards fractionation is required to move
                               the liquid towards the target composition.
                               For forward fractionation, temperature will be
                               reduced, by an amount equal to the absolute value
                               of ALPHAMELTS_DELTAT, each time. For backwards
                               fractionation the program will step down (by the
                               same temperature increment) until one or more
                               ‘allowed’ solid phases join the assemblage. The
                               ‘allowed’ solid phases are any MgO-brearing
                               phases that have been ‘allowed’ by setting
                               ALPHAMELTS_FRACTIONATE_SOLIDS or by having
                               ‘Fractionate:’ lines in the melts_file. The
                               routine then assimilates these phases before
                               searching for the new liquidus. For H2O-rich
                               compositions, the liquidus temperature will be
                               for a melt composition that is water-saturated,
                               but not oversaturated enough to exsolve vapor (it
                               may also help to buffer aH2O).
                               Output will be written each time the new liquidus
                               is found. Forward- or backward-fractionated trace
                               element compostions will be calculated if
                               ALPHAMELTS_DO_TRACE is on. Execution continues
                               until the target MgO or Mg# is reached or just
                               passed; therefore, the smaller ALPHAMELTS_DELTAT
                               is, the closer the liquid composition will be to
                               the target. In the limit as the step size tends
                               to zero a liquid composition that is on, say,
                               the plagioclase + clinopyroxene cotectic would
                               be expected to stay on the cotectic as back
                               fractionation proceeds (at least until a thermal
                               divide is encountered); in practice one or other
                               solid phase may occasionally be dropped from the
                               assemblage. For complicated peritectic
                               relationships the ‘Amoeba’ routine (menu option
                               17) is more useful for constraining a plausible,
                               but still non-unique, parental melt."""
        ),
    ),
    "MGO_TARGET": mvar(
        default=8.0,
        type=float,
        validator=typecast(
            float, [lambda x: isinstance(x, float), lambda x: (x > 0.0) & (x < 100.0)]
        ),
        overridden_by=["MGNUMBER_TARGET"],
        desc=description(
            """Sets the target MgO content of the liquid for forward
                       or backward fractionation to the value in wt %. When
                       using ‘Amoeba’ (menu option 17) the value of
                       ALPHAMELTS_MGO_TARGET is used to set the MgO of the
                       parental liquid composition and the target MgO content
                       for the evolved liquid composition is taken from the
                       melts_file(s). Once ‘Amoeba’ is finished, reading in
                       another melts_file will revert to using
                       ALPHAMELTS_MGO_TARGET as the stop point for forward or
                       backward fractional crystallization."""
        ),
    ),
    "MGNUMBER_TARGET": mvar(
        set=False,
        type=float,
        validator=typecast(
            float, [lambda x: isinstance(x, float), lambda x: (x > 0.0) & (x < 100.0)]
        ),
        desc=description(
            """Sets the target Mg# of the liquid for forward or
                            backward fractionation to the value in mol %. When
                            using ‘Amoeba’ the value of
                            ALPHAMELTS_MGNUMBER_TARGET is used in a comparable
                            way to ALPHAMELTS MGO_TARGET above."""
        ),
    ),
    "ASSIMILATE": mvar(
        set=False,
        type=bool,
        desc=description(
            """This environment variable causes a user-defined mass
                       of a second bulk composition to be added after each
                       calculation stage (see ‘Bulk composition’). It is
                       intended for calculations at specified P-T conditions
                       (e.g. isobaric, isothermal or PTpath modes) or for
                       heat-balanced assimilation in isenthalpic mode. It also
                       works for isentropic or isochoric constraints under
                       certain circumstances but these options are under
                       development and, as yet, untested. Melt may be extracted
                       or solid phases fractionated simultaneously.
                       On execution, if the mode is isothermal or
                       ALPHAMELTS_DELTAT is zero and the mode is isobaric or
                       ALPHAMELTS_DELTAP is zero then you will be asked the
                       number of iterations you wish to perform. The program
                       will then request the file type / number of files and the
                       name(s) before the first equilibration9. The assimilant
                       bulk composition may be fixed by a single enrichment_file
                       or binary restart_file, or by providing separate
                       enrichment_files for each mineral phase in the
                       assimilant. For separate enrichment_files, phase
                       compositions are given in wt% oxides and it is up to the
                       user to ensure the solid compositions are close to
                       stoichiometric in the appropriate pure phase or solid
                       solution model (see the forum for a list of mineral phase
                       end members). Trace element concentrations should be the
                       bulk assimilant values rather than individual mineral
                       ones. A single liquid composition can be input instead
                       of, or in addition to, mineral compositions.
                       Alternatively, multiple enrichment_files can be requested
                       so that the assimilant bulk composition can be changed
                       for each iteration. To use multiple enrichment_files in
                       this way, see the ‘Table file’ section for the filename
                       format. The program assumes that the indices are reset
                       each time option 4 is called.
                       You can enter the mass of assimilant to be added at each
                       subsequent stage or, for melts_file(s) only, take the
                       value(s) from the ‘Initial Mass:’ line(s). If the mass of
                       assimilant is specified for separate mineral melts_files
                       then the value entered is the total mass and the
                       ‘Initial Mass:’ lines will be used to determine the
                       proportions of the mineral phases. Major and, if
                       appropriate, trace elements will be mixed (to mix trace
                       elements only use ALPHAMELTS_FLUX_MELTING). If trace
                       element calculations are switched on, the
                       enrichment_file(s) must contain the same trace elements
                       as the melts_file. For separate mineral files, the trace
                       elements can be included in each mineral file, to avoid
                       read errors, but only the vales from the first
                       enrichment_file will be used. If running in isothermal
                       mode, or the like, then the P, T and fO2 buffer from the
                       assimilant file will be ignored and set to the current
                       values, regardless of the input file type. For
                       isenthalpic mode: please set ALPHAMELTS_DELTAP to zero,
                       for reasons given in ‘Thermodynamic Path’, and
                       ALPHAMELTS_DELTAT to zero so that the number of
                       iterations can be controlled. We suggest you set
                       ALPHAMELTS_SAVE_ALL so that you can gradually build up
                       the number of iterations. In this mode, if you use a
                       single text enrichment_file (or files) for the assimilant
                       then the program will try to find a thermodynamically
                       equilibrated state, using a superliquidus start. If you
                       use separate mineral phase enrichment_files the
                       enthalphy of each phase is calculated at temperature
                       given in the first enrichment_file, without
                       re-equilibration; this is similar to the GUI
                       assimilation routines. If you use a binary assimilant
                       file then it will use the previously calculated starting
                       conditions but the pressure in the file must match the
                       current pressure (or the temperatures must match for
                       isochoric mode). If this is not the case an error message
                       will be printed; either (a) redo the restart_file for the
                       correct conditions or (b) use menu option 2 to set the
                       current values to those used to generate the restart_file
                       and then call option 4 again."""
        ),
    ),
    "FLUX_MELTING": mvar(
        set=False,
        type=bool,
        desc=description(
            """This variable causes a user-defined proportion of a
                         second composition, trace elements only, to be mixed
                         in after each calculation stage (see ‘Bulk
                         composition’). If used in conjunction with
                         ALPHAMELTS_CONTINUOUS_MELTING and
                         ALPHAMELTS_DO_TRACE_H2O it can simulate flux melting by
                         a hydrous fluid (to simulate fluxing by a metasomatic
                         melt use ALPHAMELTS_ASSIMILATE instead). We suggest you
                         set ALPHAMELTS_SAVE_ALL so that you can gradually build
                         up the number of iterations, e.g. until the system
                         achieves steady state, by repeated calling of the
                         ‘execute’ menu option. Execution is essentially the
                         same as for ALPHAMELTS_ASSIMILATE but for trace
                         elements only. As the mass of the enriching
                         composition is not necessarily defined, the user is
                         asked instead for the mass proportion of the new
                         composition to add (similar to ‘Source Mixer’ in menu
                         option 12). Alternatively, for melts_files(s), the
                         ‘Initial Mass:’ lines may be used; in which case the
                         old and new compositions are mixed in the ratio
                         old_mass:new_mass. Note that, as major elements are not
                         mixed, the total system mass (i.e. old_mass) will be
                         unaffected."""
        ),
    ),
    "DRY_ITER_PATIENCE": mvar(
        default=100,
        type=int,
        validator=typecast(
            int, [lambda x: isinstance(x, int), lambda x: (x >= 0) & (x <= 100)]
        ),
        desc=description(
            """If simulating flux melting or assimilation,
                              this is the maximum number of consecutive
                              iterations that alphaMELTS will run without any
                              melting occurring before it will give up and
                              return to the menu."""
        ),
    ),
    # pHMELTS and Trace Elements
    "DO_TRACE": mvar(
        set=False,
        type=bool,
        desc=description(
            """Implements attached trace element partitioning function
                     for those elements listed in the melts_file."""
        ),
    ),
    "DO_TRACE_H2O": mvar(
        set=False,
        type=bool,
        desc=description(
            """For the case where water is to be treated as a trace
                         element this option adds an iteration on the H2O
                         content, as described in Asimow, et al. [2004]."""
        ),
    ),
    "HK_OL_TRACE_H2O": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default the Mosenfelder, et al. [2005] model
                            for water solubility in olivine is used. This
                            environment variable uses the Hirth and Kohlstedt
                            [1996] model instead, which gives lower solubility
                            and consequently lower partition coefficients."""
        ),
    ),
    "HK_PXGT_TRACE_H2O": mvar(
        default="mineral-melt",
        type=bool,
        validator=lambda x: x in ["mineral-melt", "mineral-mineral"],
        desc=description(
            """If the Mosenfelder, et al. [2005] model for
                              water solubility in olivine is used then by
                              default the mineral-melt partition coefficients
                              for water with orthopyroxene, clinopyroxene and
                              garnet are still those of Hirth and Kohlstedt
                              [1996]. Setting this variable to ‘mineral-mineral’
                              means the solubility of water in opx, cpx and
                              garnet are linked to the newer water solubility
                              in olivine model, which is equivalent to
                              preserving the mineral-mineral water partition
                              coefficients of Hirth and Kohlstedt [1996].
                              Setting it to ‘mineral-melt’ retains the default
                              behavior."""
        ),
    ),
    "2X_OPX_TRACE_H2O": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default the solubility of water in
                             clinopyroxene is twice that in orthopyroxene,
                             based on the results of Hirth and Kohlstedt [1996].
                             If this variable is set then the solubility of
                             water in opx is scaled up by to be equal to the
                             cpx value, consistent with the observations of
                             Hauri et al. [2006]. This option can be used
                             regardless of whether the Mosenfelder, et al.
                             [2005] or Hirth and Kohlstedt [1996] models are
                             being used for olivine or opx, cpx and garnet. For
                             example, if either ALPHAMELTS_HK_OL_TRACE_H2O or
                             ALPHAMELTS_HK_PXGT_TRACE_H2O is set then the
                             mineral-mineral partition coefficients will be
                             those from Table 1 of Hirth and Kohlstedt [1996],
                             except that olivine-opx value of 0.2 will be
                             replaced by 0.1."""
        ),
    ),
    "TRACE_DEFAULT_DPTX": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default all partition coefficients used in
                               trace element calculations are constant. If
                               ALPHAMELTS_TRACE_VARIABLE_D is set then D =
                               D(P,T,X) are calculated for elements and phases
                               in Table 1. Constant partition coefficients will
                               be used for all other elements / phases. This
                               list may be modified in the trace_data_file, as
                               previously explained."""
        ),
    ),
    "TS_TRACE_NORMALIZATION": mvar(
        set=False,
        type=int,
        validator=typecast(
            int, [lambda x: isinstance(x, int), lambda x: (x >= 1) & (x <= 4)]
        ),
        desc=description(
            """If set, this chooses one of four
                                   compositions to normalize trace elements to
                                   (if any):
                                   1. PM Sun and McDonough [1989];
                                   2. DMM Workman and Hart [2005];
                                   3. PM McKenzie and O'Nions [1991; 1995];
                                   4. DM of McKenzie and O'Nions [1991; 1995].
                                   Sample input files showing concentrations for
                                   each of the idealized source compositions
                                   above are provided as illustrations; note
                                   that some elements (i.e. Ni, Cr, and Mn)
                                   appear as major and trace elements because
                                   their inclusion in the liquid calibration
                                   differs between MELTS and pMELTS. Isotopes
                                   are normalized to the ‘non-isotope’ abundance
                                   (see the ‘MELTS file’ section). This option
                                   is useful if the source composition given in
                                   the melts_file is different from the four
                                   options above. If you wish to normalize to
                                   the source in the melts_file the simplest
                                   thing is to provide the list of elements with
                                   ‘1.0’ for each of the abundances (except
                                   isotopes)."""
        ),
    ),
    "TRACE_INPUT_FILE": mvar(
        set=False,
        desc=description(
            """Gives the name of the trace_data_file, which may
                             be used to change partition coefficients, determine
                             whether variable partition coefficients are
                             calculated and with which parameters, as described
                             above."""
        ),
    ),
    "TRACE_USELIQFEMG": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default the Mg# of the melt, needed to
                             estimate D(P,T,X) for clinopyroxene, is estimated
                             from the clinopyroxene composition using Equation
                             35 from Wood and Blundy [1997]). If this
                             environment variable is set then the Mg# of the
                             melt is taken directly from the (pH)MELTS
                             calculated liquid composition. If no liquid is
                             present then the program will revert to the default
                             behavior and use the clinopyroxene composition."""
        ),
    ),
    # Input options
    "ALPHAMELTS_ADIABAT_BIN_FILE": mvar(
        set=False,
        desc=description(
            """Unavoidable changes within the
                                        programs mean that binary restart_files
                                        created with version 2.0+ of Adiabat_1ph
                                        will not read into alphaMELTS. Once you
                                        have read in the file, if you
                                        immediately save it again it will be
                                        updated to Adiabat_1ph 3.0 format.
                                        Adiabat_1ph 3.0 format is compatible
                                        with alphaMELTS 1.0+ so unset this
                                        environment variable for subsequent
                                        runs. For more details see
                                        ADIABAT_V20_BIN_FILE in the
                                        Adiabat_1ph 3 documentation."""
        ),
    ),
    # Output options
    "CELSIUS_OUTPUT": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default, temperature input to alphaMELTS is in
                           °C, whereas temperature output is in Kelvin. When
                           this environment variable is set, text file and
                           screen output is also in °C."""
        ),
    ),
    "SAVE_ALL": mvar(
        set=False,
        type=bool,
        desc=description(
            """By default, the main output file is only written for
                     calculations made in the most recent call to menu option 4.
                     If this variable is set then results from all calculations
                     are saved and output. This provides a simple way to record
                     single calculations (menu option 3) or to build up results
                     from multiple iterations or multiple melts_files. Note that
                     even if the appropriate environment variables are set, no
                     solid or liquid fractionation will occur until menu option
                     4 is run. If using ‘Amoeba’ (menu option 17) the
                     ALPHAMELTS_SAVE_ALL function will be temporarily switched
                     off (so that iterations of the search algorithm are not
                     saved); when it is finished a single calculation (menu
                     option 3) may be used to write output for the best-fit
                     parental melt."""
        ),
    ),
    "SKIP_FAILURE": mvar(
        set=False,
        type=bool,
        desc=description(
            """Normally failure of the minimization routines means
                         that the alphamelts executable must be restarted to
                         clear the memory. If this environment variable is set
                         then a copy of the thermodynamic state is made each
                         time a successful calculation is made. If the next
                         calculation fails, this last good state (or the bulk
                         composition and starting conditions from the
                         melts_file, if it is the first calculation) is used
                         for subsequent attempts. This may be useful if you are
                         very close to a reaction when the algorithms may have
                         trouble deciding whether to add a phase to the
                         assemblage and need to overstep just slightly. It is
                         also helpful when trying out starting solutions using
                         options 2 and 3. If ALPHAMELTS_SAVE_ALL is also set,
                         the last good state will also be written to the output
                         files as a placeholder; it should be obvious that this
                         represents a skipped failure as all values, including
                         pressure and temperature, will be identical to the
                         previous output.."""
        ),
    ),
    "FAILED_ITER_PATIENCE": mvar(
        default=10,
        type=int,
        validator=typecast(
            int, [lambda x: isinstance(x, int), lambda x: (x >= 0) & (x <= 10)]
        ),
        desc=description(
            """If the minimization routines do not recover
                                 in the next few iterations then one drawback of
                                 ALPHAMELTS_SKIP_FAILURE is that it can lead to
                                 infinite loops of failure or, if triggered
                                 repeatedly, eventually to a segmentation fault.
                                 If ALPHAMELTS_SKIP_FAILURE is set, this
                                 variable is the maximum number of consecutive
                                 failed iterations that alphaMELTS will run
                                 before a clean return to the menu where
                                 parameters, such as the choice of fO2 buffer,
                                 can be adjusted or alphaMELTS completely
                                 restarted. If running in PTgrid or TPgrid
                                 modes, it will be the maximum number of failed
                                 iterations before alpahMELTS moves to the next
                                 internal loop (T-loop or P-loop
                                 respectively)."""
        ),
    ),
    "INTEGRATE_FILE": mvar(
        set=False,
        type=bool,
        desc=description(
            """Normally the integrated_output_file can be written
                           immediately after execution finishes, but if
                           alphamelts tends to crash before the menu option 16
                           can be called then memory contents will be lost. As a
                           precaution, ALPHAMELTS_INTEGRATE_FILE should be set
                           to record details of the packets of melt extracted
                           during melting to a file. If run_alphamelts.command
                           is restarted with the same settings and menu option
                           16 is called before execution then the resulting file
                           will be used as input for the melt integration. Note
                           that ALPHAMELTS_INTEGRATE_FILE is not the
                           integrated_output_file itself but an intermediate
                           file that may be used in its generation."""
        ),
    ),
    "LATENT_HEAT": mvar(
        set=False,
        type=bool,
        desc=description(
            """Normally text file output is done just after
                        equilibration and before any fractionation. If
                        ALPHAMELTS_LATENT_HEAT to set to ‘true’ alphamelts will
                        also output the state just after fractionation and just
                        before the next equilibration (when the temperature has
                        been lowered and the system cooled but not allowed to
                        crystallise any more yet). This gives a simple way to
                        calculate the latent heat of crystallization, without
                        having to do the cooling adjustment described in Ghiorso
                        [1997]. Note that, for the same MORB composition, the
                        results will differ from Figure 5 of Ghiorso [1997].
                        This apparent discrepancy is due to changes in the MELTS
                        software since it was first released and not to
                        differences in how the variation in latent heat is
                        accessed."""
        ),
    ),
    "QUICK_OUTPUT": mvar(
        set=False,
        type=bool,
        desc=description(
            """If this environment variable is set then
                         equilibrated thermodynamic states from previous
                         calculations are not saved, except as required for
                         ALPHAMELTS_SKIP_FAILURE etc. For calculations
                         comprising a large number of iterations, such as those
                         on a closely spaced P-T grid, setting this variable can
                         significantly reduce memory usage and therefore
                         increase computational speed as the calculation
                         proceeds. Reporting of fO2 will be affected slightly.
                         The Phase_mass_tbl.txt and Phase_vol_tbl.txt files will
                         not be written, which will cause
                         run_alphamelts.command to issue a warning about not
                         being able to find these files; the warning can be
                         safely ignored."""
        ),
    ),
    # Others
    "MULTIPLE_LIQUIDS": mvar(
        set=False,
        type=bool,
        desc=description(
            """This turns on exsolution of immiscible liquids. The
                             solvi are not very well determined, and we do not
                             recommend serious use of this feature, but in some
                             cases operation inside an unrecognized two-liquid field
                             can lead to path-dependent non-unique equilibria. This
                             option should not be used if trace element calculations
                             are enabled."""
        ),
    ),
    "FRACTIONATE_SECOND_LIQUID": mvar(
        set=False,
        type=bool,
        desc=description(
            """When running with
                                      ALPHAMELTS_MULTIPLE_LIQUIDS set, treat all
                                      liquids except the first as fractionating
                                      phases and remove them from the system
                                      after each equilibration."""
        ),
    ),
    "FOCUS": mvar(
        set=False,
        type=bool,
        desc=description(
            """Option to do the focusing calculation described in Asimow
                  and Stolper [1999]. It works by multiplying the mass of liquid
                  in system by a fixed factor after each equilibration."""
        ),
    ),
    "FOCUS_FACTOR": mvar(
        set=False,
        type=float,
        validator=typecast(float, numeric_validator),
        desc=description(
            """When ALPHAMELTS_FOCUS is set, this determines the
                         multiplication factor for the mass of liquid. Usually
                         it will be a number slightly greater than unity, like
                         the 100th root of 2. If ALPHAMELTS_FOCUS is set without
                         a value in ALPHAMELTS_FOCUS_FACTOR the focusing
                         calculation loop will be ignored."""
        ),
    ),
    "INTEGRATE_PHI": mvar(
        set=False,
        type=bool,
        desc=description(
            """Normally, packets of extracted melt are saved in
                          memory (and, optionally, written out to the
                          ALPHAMELTS_INTEGRATE_FILE ‘crash’ file, described
                          above) by mass i.e. in terms of dF. If this variable
                          is set then the packets of extracted melt will be
                          saved by volume i.e. in terms of dPhi. As the full
                          pressure integral (calculation of crustal thickness
                          etc.) is not strictly correct for increments in Phi,
                          that option will not be offered. However, the
                          opportunity to interpolate for equal increments of
                          melt fraction by volume will be available."""
        ),
    ),
}
