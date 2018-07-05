## Melts Web Services

These services provide the ability to conduct melting and fractionation computations using whole-rock major element compositions. Some information can be found here: [website](http://melts.ofm-research.org/web-services.html).

### Input

The MELTS WS Compute web service can be found at [http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/Compute](http://thermofit.ofm-research.org:8080/multiMELTSWSBxApp/Compute).

```python

requests.POST(contentType='text/xml',
              dataType='xml',
              data=<insertXMLinputstring>)
```

The xml input has the general structure:
```xml
<?xml version'1.0' encoding'UTF-8'?>
  <MELTSinput>
      <initialize>
          <SiO2>48.68</SiO2>
          ...
          <K2O>0.03</K2O>
          <P2O5>0.08</P2O5>
          <H2O>0.20</H2O>
      </initialize>
      <calculationMode>equilibrate</calculationMode>
      <title>Enter a title for the run</title>
      <constraints>
        <setTP>
          <initialT>1200</initialT>
          <initialP>1000</initialP>
        </setTP>
      </constraints>
  </MELTSinput>
```

The web service has a memory: "The web service remembers the state of the calculation as long as cookies are enabled on the client side and repeated calls to the service do not specify an "initialize" block in the input XML. Each call to the web service that contains an <initialize/> tag will reinitialize the server state, discarding results of previous calculations."

#### Parameters
This information is principally from the [XML schema](http://melts.ofm-research.org/WebServices/MELTSinput.xsd) and [documentation](http://melts.ofm-research.org/WebServices/MELTSinput_Schema_Generated_Docs/MELTSinput.html) on the melts website.
* `initialize`
  * `modelSelection`
    * *`MELTS_v1.0.x`* |, *`MELTS_v1.1.x`* | *`MELTS_v1.2.x`* | *`pMELTS_v5.6.1`*
    * all compositional variables
* `fractionateOnly`: *`fractionateSolids`*, *`fractionateFluids`*, *`fractionateLiquids`*  (choose 1-2)

* `constraints` (choose one; thermoengine also has SV)
  * These modes are available:
    1. *`setTP`* temperature-pressure
    2. *`setTV`* temperature-volume
    3. *`setHP`* enthalpy-pressure
    4. *`setSP`* entropy-pressure
  * `initial<var>` must be set
  * Optional: `final<var>`, `inc<var>`, `d<var2>d<var1>`
  * `fo2Path`: *`none`* | *`fmq`* | *`coh`* | *`nno`* | *`iw`* | *`hm`*

* `fractionationMode`: *`fractionateNone`*, *`fractionateSolids`*, *`fractionateFluids`*, *`fractionateLiquids`* (choose 0-2)

* `multLiquids`: `True` | `False`
* `suppressPhase`: `str`
* `assimilant`
  * `temperature`
  * `increments`: `int`
  * `mass`
  * `units`: `vol` | `wt`s
  * `phase` (any number)
    * `amorphous` | `solid` | `liquid` (with properties..)

### Output

"The output from the web service is in an XML string, structured according to the schema. If you prefer to work with output in JSON format, you can use a Javascript function to convert the XML to JSON (as shown in the example referenced below), or you can make the conversion in whatever language or platform you use to access the web service, such as Python or R.""

The (format-modified) output is shown below:
```JSON
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

```
