# Normalisation

A selection of reference compositions are included:

```python
>>> from pyrolite.normalisation import ReferenceCompositions
>>> refcomp = ReferenceCompositions()
{
'Chondrite_PON': Model of Chondrite (Palme2014),
'D-DMM_WH': Model of DepletedDepletedMORBMantle (Workman2005),
'DMM_WH': Model of DepletedMORBMantle (Workman2005),
'DM_SS': Model of DepletedMantle (Salters2004),
'E-DMM_WH': Model of EnrichedDepletedMORBMantle (Workman2005),
'PM_PON': Model of PrimitiveMantle (Palme2014)
}
```

```python
>>> CH = refcomp['Chondrite_PON']
>>> PM = refcomp['PM_PON']
>>> CH[REE()]
      value  unc_2sigma units
var                           
La    0.2414    0.014484   ppm
Ce    0.6194    0.037164   ppm
...
Tm   0.02609    0.001565   ppm
Yb    0.1687    0.010122   ppm
Lu   0.02503    0.001502   ppm
```

The `normalize` method can be used to normalise dataframes to a given reference (e.g. for spiderplots):
```python
>>> from pyrolite.plot import spiderplot
>>> refcomp = ReferenceCompositions()
>>> CH = refcomp['Chondrite_PON']
>>> DMM = refcomp['DMM_WH']
>>>
>>> df = DMM.data.loc[REE(), ['value']]
>>> spiderplot(CH.normalize(df), label=f'{DMM.Reference}')
```

<img src="https://raw.githubusercontent.com/morganjwilliams/pyrolite/develop/docs/resources/SpiderplotExample.png" alt="SpiderPlotExample" height="250px"/>

More reference compositions will soon be included (e.g. Sun and McDonough, 1989).
