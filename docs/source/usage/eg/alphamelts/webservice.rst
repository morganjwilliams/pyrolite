Interfacing with the Melts Web Services
=========================================

The MELTS web services provide the ability to conduct melting and fractionation
computations using whole-rock major element compositions. Some information can be found
`here <http://melts.ofm-research.org/web-services.html>`__. The MELTS WS Compute web
service can be found
`here <http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/Compute>`__.  A minimal
interface to this web service is provided in :mod:`pyrolite.ext.alphamelts.web`.
The basic functionality of this is demonstrated below (obtaining valid phases and oxides
for a specific model of MELTS, and performing a single computation). Here we use a
dictionary to pass information to the service; this can be customised to your use case.

.. literalinclude:: ../../../../_examples/alphamelts/meltsweb.py
  :language: python
  :end-before: # %% Oxides

To obtain a list of phases or oxides for a specific model of MELTS (defaulting to
rhyolite-MELTS version 1.0.2, can be changed with a `modelSelection` parameter; see
below), you can use the Phases and Oxides services:

.. literalinclude:: ../../../../_examples/alphamelts/meltsweb.py
  :language: python
  :start-after: # %% Oxides
  :end-before: # %% Phases

.. code-block:: python

  ["SiO2", "TiO2", "Al2O3", "Fe2O3", "Cr2O3", "FeO", "MnO", "MgO", "NiO", "CoO", "CaO",
      "Na2O", "K2O", "P2O5", "H2O", "CO2",  "SO3", "Cl2O-1", "F2O-1"]

.. literalinclude:: ../../../../_examples/alphamelts/meltsweb.py
  :language: python
  :start-after: # %% Phases
  :end-before: # %% Compute

.. code-block:: python

  ["olivine", "fayalite", "sphene", "garnet", "melilite", "orthopyroxene",
   "clinopyroxene", "aegirine", "aenigmatite", "cummingtonite", "amphibole",
   "hornblende", "biotite", "muscovite", "feldspar", "quartz", "tridymite",
   "cristobalite", "nepheline", "kalsilite", "leucite", "corundum", "sillimanite",
   "rutile", "perovskite", "spinel", "rhm-oxide", "ortho-oxide", "whitlockite",
   "apatite", "water", "alloy-solid", "alloy-liquid"]'

The compute service is also simple to access, and can be customised to provide
different versions of MELTS and different computation modes
(:code:`"findLiquidus", "equilibrate", "findWetLiquidus"`).

.. literalinclude:: ../../../../_examples/alphamelts/meltsweb.py
  :language: python
  :start-after: # %% Compute

Compute Parameters
-------------------

Parameters can be passed to the compute query to customise the calculation. A
selection of parameters and possible values can be found in the
`XML schema <http://melts.ofm-research.org/WebServices/MELTSinput.xsd>`__ and
`documentation <http://melts.ofm-research.org/WebServices/MELTSinput_Schema_Generated_Docs/MELTSinput.html>`__
on the melts website.

..
  :code:`initialize`

    * :code:`modelSelection`

      * *`MELTS_v1.0.x`* |, *`MELTS_v1.1.x`* | *`MELTS_v1.2.x`* | *`pMELTS_v5.6.1`*
      * all compositional variables

  :code:`fractionateOnly`:

    * *`fractionateSolids`*, *`fractionateFluids`*, *`fractionateLiquids`*  (choose 1-2)

  :code:`constraints` (choose one; thermoengine also has SV)

    * These modes are available:

      1. *`setTP`* temperature-pressure
      2. *`setTV`* temperature-volume
      3. *`setHP`* enthalpy-pressure
      4. *`setSP`* entropy-pressure

    * :code:`initial<var>` must be set
    * Optional: :code:`final<var>`, :code:`inc<var>`, :code:`d<var2>d<var1>`
    * :code:`fo2Path`: *`none`* | *`fmq`* | *`coh`* | *`nno`* | *`iw`* | *`hm`*

  :code:`fractionationMode`: *`fractionateNone`*, *`fractionateSolids`*, *`fractionateFluids`*, *`fractionateLiquids`* (choose 0-2)

  :code:`multLiquids`: :code:`True` | :code:`False`

  :code:`suppressPhase`: `str`

  :code:`assimilant`
    * :code:`temperature`
    * :code:`increments`: :code:`int`
    * :code:`mass`
    * :code:`units`: `vol` | `wt`s
    * :code:`phase` (any number)

      * `amorphous` | `solid` | `liquid` (with properties..)

Compute Output
-------------------

An example of formatted JSON output from the compute service is shown below.

.. code-block:: json

    { "status":"Success: Equilibrate",
      "sessionID":"552291051.596800.1804289383",
      "title":"Enter a title for the run",
      "time":"Mon Jul 2 23:10:51 2018",
      "release":"MELTS Web Services, MELTS v.1.0.x",
      "buildDate":"Sep 27 2016",
      "buildTime":"08:37:35",
      "temperature":"1200",
      "pressure":"1000",
      "log_fO2":"-9.292508501350249972",
      "deltaHM":"-6.5843402903737722198",
      "deltaNNO":"-1.7295882284656141081",
      "deltaFMQ":"-1.0655143052398710068",
      "deltaCOH":"1.1140739919464248686",
      "deltaIW":"2.4523554636227675729",
      "liquid":{"mass":"79.224930087117130029",
                "density":"2.6888739764023834589",
                "viscosity":"2.1808546355527487215",
                "gibbsFreeEnergy":"-1281817.8163060888182",
                "enthalpy":"-967241.03052840172313",
                "entropy":"213.54022725295251917",
                "volume":"29.463980380782754054",
                "dvdt":"0.0022228926591244097498",
                "dvdp":"-0.00018477679830941845605",
                "d2vdt2":"6.0547123990430293847e-08",
                "d2vdtdp":"-1.9823487546973320295e-08",
                "d2vdp2":"8.2176933278351927828e-09",
                "heatCapacity":"117.54760903391918703",
                "SiO2":"49.070879713300428193",
                "TiO2":"1.2746984706149677713",
                "Al2O3":"15.547239314502563801",
                "Fe2O3":"1.1219160810065404998",
                "Cr2O3":"0.050467309333368480517",
                "FeO":"8.6472380250776375021",
                "MgO":"8.6112592267950649472",
                "CaO":"12.433184035995937577",
                "Na2O":"2.8525803875263218146",
                "K2O":"0.037113330969942535942",
                "P2O5":"0.10097831567919893225",
                "H2O":"0.25244578919801691219"},
      "solid":[{"name":"olivine",
                "formula":"(Ca0.01Mg0.85Fe''0.15Mn0.00Co0.00Ni0.00)2SiO4",
                "mass":"5.0611455532498279553",
                "density":"3.256264076719213918",
                "gibbsFreeEnergy":"-80793.633565014824853",
                "enthalpy":"-62670.686010467339656",
                "entropy":"12.302173950071260577",
                "volume":"1.5542798231367915829",
                "dvdt":"7.3368482662024503633e-05",
                "dvdp":"-1.159590122970580791e-06",
                "d2vdt2":"2.5826647116800414035e-08",
                "d2vdtdp":"2.9194839589241691073e-14",
                "d2vdp2":"3.4022602849075366441e-12",
                "heatCapacity":"6.3269887937074473783",
                "SiO2":"39.907476858520290364",
                "FeO":"14.58265239333134744",
                "MgO":"44.973873816349971833",
                "CaO":"0.53599693179839225099",
                "component":[{"name":"tephroite",
                              "formula":"Mn2SiO4",
                              "moleFraction":"0"},
                              {"name":"fayalite",
                               "formula":"Fe2SiO4",
                               "moleFraction":"0.15279468642281898716"},
                              {"name":"co-olivine",
                               "formula":"Co2SiO4",
                               "moleFraction":"0"},
                               {"name":"ni-olivine",
                                "formula":"Ni2SiO4",
                                "moleFraction":"0"},
                               {"name":"monticellite",
                                "formula":"CaMgSiO4",
                                "moleFraction":"0.014390161918421261189"},
                               {"name":"forsterite",
                                 "formula":"Mg2SiO4",
                                 "moleFraction":"0.83281515165875974471"}]
                                 },
