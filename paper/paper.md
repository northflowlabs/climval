---
title: 'climval: aligned, recorded validation of climate model output for Pangeo workflows and continuous integration'
tags:
  - Python
  - climate
  - model evaluation
  - xarray
  - dask
  - CMIP6
  - Pangeo
authors:
  - name: Tom Danny Pedersen
    affiliation: 1
    corresponding: true
affiliations:
  - name: Northflow Technologies AS, Norway
    index: 1
date: 30 October 2026
bibliography: paper.bib
---

# Summary

Climate models are checked against observations and reanalyses all the time:
when a model is developed, when its output is post-processed, when it is
downscaled, and when it is chosen for an impact study. Each check compares two
gridded datasets, and before any statistic can be computed the two have to be
put on the same points. They usually differ in grid, longitude convention,
calendar, time step, units and missing values. climval is a Python library
that does this alignment explicitly and in a fixed order, computes weighted
comparison metrics with xarray [@hoyer2017], lazily on dask arrays
[@rocklin2015], and records every step with the result. A configuration file
and a command, `climval check`, turn the same comparison into a pass or fail
gate for continuous integration, and a GitHub Action runs that gate in a
workflow.

# Statement of need

Model evaluation code is often written again for each study. The metric is
the easy part; the errors sit in the preparation. A reanalysis on a
-180 to 180 longitude grid compared with a model on 0 to 360, a `360_day`
model calendar matched to a Gregorian calendar by timestamp, a difference of
273.15 between K and degC, missing values present in one dataset and not the
other, or a spatial mean without area weights each change the result, and
none of them raises an error. When the preparation is not recorded, the
number cannot be checked by a reader or reproduced by another group.

climval addresses this for one specific task: the comparison of a candidate
dataset with a reference dataset. Its audience is scientists and engineers
who already work with xarray and the Pangeo stack [@abernathey2021] and need
a validation step they can put into a notebook, a processing pipeline or a
continuous integration job: model developers who want a regression test for
a new configuration, data engineers who check post-processed or downscaled
output, and researchers who compare a few CMIP6 models [@eyring2016] with
ERA5 [@hersbach2020] for a region.

# State of the field

Several mature tools evaluate climate models. ESMValTool [@righi2020;
@eyring2020] runs recipe-driven diagnostics across CMIP ensembles, with CMOR
fixes, a large observational archive and figures for assessment reports. The
PCMDI Metrics Package [@lee2024] computes a standard set of performance
metrics for CMIP models, building on the summary statistics of
@gleckler2008. ILAMB [@collier2018] benchmarks land models against a curated
collection of observations with a scoring system. These tools are built for
campaigns: many models, many variables, a fixed archive and a configuration
step before a run. xskillscore [@xskillscore] provides deterministic and
probabilistic verification metrics on xarray objects that are already
aligned.

climval sits between these. It is not a diagnostic suite and does not try to
replace one: it has no recipes, no observational archive and no CMOR fixes.
It is smaller than the campaign tools and does more than a metric library: it
takes two datasets as they come, aligns them, records how, and returns
weighted metrics on the aligned pair. We built it rather than contributing to
an existing tool because the missing piece was a light, typed step that works
directly on lazy xarray objects from cloud storage, installs with pip, and
answers with an exit code in a CI job. Adding these properties to a campaign
tool would change its design; adding alignment to a metric library would
widen its scope beyond verification. For the metrics both libraries define
(RMSE, MAE, mean error, Pearson and Spearman correlation), climval's test
suite checks agreement with xskillscore to a relative tolerance of 1e-10, so
users can move between the two.

# Software design

The library has three layers. `climval.align` runs the alignment steps in a
fixed order: input resolution, CF-aware coordinate detection, the target grid,
longitude convention, region selection (including boxes across the
antimeridian and IPCC AR6 regions [@iturbide2020] through regionmask
[@regionmask]), time alignment, regridding (bilinear or nearest with xarray,
conservative through xESMF [@xesmf]), unit conversion, a paired
missing-value mask and cell-area weights computed from the grid's cell
bounds. By default the comparison grid is the coarser of the two, so neither
input is interpolated to a finer grid than it has. Time is aligned by
calendar labels, not by timestamps: each input's native frequency is read
from its time bounds, a `frequency` attribute or its time steps (climval
raises when these cannot tell it), the finer input is aggregated to the
common frequency, a period counts only where enough of its native steps are
valid, and periods are matched by year and month, which makes `noleap`,
`360_day` and Gregorian calendars comparable at monthly resolution. Each
step appends a record, so the alignment can be read back.

`climval.compare` evaluates metrics on the aligned pair over named samples:
all points in space and time, the spatial pattern of the time-mean fields
(the statistic behind the Taylor diagram [@taylor2001]), the pattern at each
time step, and the time series at each grid point, optionally grouped by
season, month or year. Monthly time steps are weighted by their length in
days, and seasons are contiguous by default: a December belongs to the
following winter, and incomplete winters at the ends of the period are left
out. The metrics include mean bias, mean absolute error,
root mean square error and its centred form, pattern correlation, the
standard deviation ratio and the skill score of @taylor2001, the Kling-Gupta
efficiency [@gupta2009] and percentile biases. Every metric is written as a weighted xarray
reduction, so dask arrays stay lazy until one compute pass evaluates all
metrics and samples together; percentiles of samples larger than one dask
chunk are estimated in the same pass from weighted histograms. Extra dimensions such as an ensemble member or
a model index are kept, so one call scores a whole ensemble. The result
carries a provenance record with software versions, input identity
(including CMIP attributes such as `source_id` and `tracking_id`), every
alignment step and hashes of the aligned values.

`climval check` reads YAML, TOML or JSON. Each check names a reference and a
candidate (a path, a URL or an intake-esm [@intakeesm] catalogue query), the
alignment options and thresholds on metric values for chosen samples and
groups. The whole file is validated before any data is opened. The command
writes a console summary, a JSON result, a Markdown summary suitable for a
GitHub job summary and an HTML report, and exits with 0 when every threshold
holds, 1 when a threshold fails and 2 when the configuration or the data
cannot be used. A composite GitHub Action in the repository installs climval,
runs the gate, writes the job summary and uploads the report.

The main trade-offs are scope and defaults. climval compares two datasets at
a time; multi-model studies run one comparison per model, or align the
models onto one grid and stack them along a dimension. Alignment choices
that change results, such as the target grid, the regridding method, the
coverage threshold and the season definition, are explicit options with
documented defaults, and the values used are recorded. When the data cannot
decide a choice, such as the frequency of a single time step, climval raises
instead of guessing. The library never substitutes synthetic data for
missing inputs.

# Example

```python
import climval

result = climval.compare(
    era5,                       # reference: xarray object, path or URL
    cmip6_tas,                  # candidate, for example from intake-esm
    variable=("t2m", "tas"),
    region=climval.Box(lat=(35, 72), lon=(-10, 40)),
    period=("1995", "2014"),
    by="season",
)
result.to_dataframe()           # sample, group, metric, value, units
result.check({"pearson_r": {"min": 0.95, "sample": "climatology"}}).passed
```

The documentation includes two notebooks that run on public cloud data
without credentials: ERA5 from the analysis-ready ARCO-ERA5 store
[@carver2023] against five CMIP6 historical simulations from the Pangeo CMIP6
collection over Europe, each compared on its own grid, the coarser one, with
a Taylor diagram per season, and the CI gate applied to two of these models
over the AR6 Northern Europe land region.

# Research impact statement

climval 0.2.0 is new, and we do not yet know of publications that use it.
Its near-term value rests on what can be checked now: the alignment steps
and metrics are covered by tests, including agreement with xskillscore and
end-to-end comparisons against manually aligned data; the examples run on
public ERA5 and CMIP6 data and store their outputs; and the GitHub Action is
exercised in the repository's own CI on committed test data. The package is
on PyPI under the Apache License 2.0, with documentation at
<https://northflowlabs.github.io/climval/>.

# AI usage disclosure

Generative AI (Anthropic's Claude) was used to write parts of the code, the
tests, the documentation, the example notebooks and a first draft of this
paper, under the direction of the author. Its output was checked by the test
suite, by executing the example notebooks on real data and comparing every
statement in their text with the stored outputs, by an independent review of
the alignment engine, and by the author's review.

# Acknowledgements

We acknowledge the World Climate Research Programme, which, through its
Working Group on Coupled Modelling, coordinated and promoted CMIP6, the
climate modelling groups for producing and making available their model
output, the Earth System Grid Federation for archiving the data and providing
access, and the funding agencies who support CMIP6 and ESGF. We thank the
Pangeo community for the cloud CMIP6 collection and Google for ARCO-ERA5. The
examples contain modified Copernicus Climate Change Service information.
climval is developed and maintained by Northflow Technologies AS.

# References
