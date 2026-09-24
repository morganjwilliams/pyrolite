```{raw} latex
\chapter{Introduction}
```

# pyrolite

pyrolite is a set of tools for making the most of your geochemical data.

The python package includes functions to work with compositional data, to transform
geochemical variables (e.g. elements to oxides), functions for common plotting
tasks (e.g. spiderplots, ternary diagrams, bivariate and ternary density diagrams),
and numerous auxiliary utilities.

- On this site you can browse the [API](./api/API.md), or look
  through some of the [usage examples](./examples/index.md).
- There's also a quick [installation guide](./installation.md), a list of
  [recent changes](./dev/changelog.md) and some notes on
  where the project is heading in the [roadmap](./dev/roadmap.md).
- If you're interested in [contributing to the project](./dev/contributing.md), there are
  many potential avenues, whether you're experienced with python or not.

:::{note}
pyrolite has been
[published in the Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.02314),
and a recent publication focusing on using `lambdas` and tetrads to parameterise
Rare Earth Element patterns has been
[published in Mathematical Geosciences](https://doi.org/10.1007/s11004-021-09959-5)!
:::

## Why *pyrolite*?

The name *pyrolite* is an opportunistic repurposing of a term used to describe an early
model mantle composition proposed by Ringwood [^cite_ringwood1962], comprised principally
of **pyr**-oxene & **ol**-ivine. While the model certainly hasn't stood the test of time,
the approach optimises the aphorism "All models are wrong, but some are useful"
[^cite_box1976]. It is with this mindset that pyrolite is built, to better enable you to
make use of your geochemical data to build and test geological models.

[^cite_ringwood1962]: Ringwood, A.E. (1962). A model for the upper mantle.
    Journal of Geophysical Research (1896-1977) 67, 857–867.
    [doi: 10.1029/JZ067i002p00857](https://doi.org/10.1029/JZ067i002p00857)

[^cite_box1976]: Box, G.E.P. (1976). Science and Statistics.
    Journal of the American Statistical Association 71, 791–799.
    [doi: 10.1080/01621459.1976.10480949](https://doi.org/10.1080/01621459.1976.10480949)

```{raw} latex
\chapter{Getting Started}
```

```{toctree}
:caption: Getting Started
:hidden: true
:maxdepth: 1

installation
gettingstarted
gallery/examples
gallery/tutorials
cite
```

```{raw} latex
\chapter{Development}
```

```{toctree}
:caption: Development
:hidden: true
:maxdepth: 1

dev/changelog
dev/roadmap
dev/conduct
dev/contributing
dev/contributors
dev/development
dev/release
```

```{raw} latex
\chapter{Reference}
```

```{toctree}
:caption: Reference
:hidden: true
:maxdepth: 1

api/API
data/index
ext/extensions
```
