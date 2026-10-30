# Units

climval converts the candidate into the reference's units before any metric
is computed, using the `units` attribute of each variable.

| Quantity | Built in |
|---|---|
| Temperature | K, degC, degF (and spellings such as `kelvin`, `degrees_C`, `deg_C`) |
| Precipitation flux | kg m-2 s-1 (same as mm s-1), mm h-1, mm day-1 |
| Pressure | Pa, hPa (mbar), kPa |
| Fractions | 1, % |
| Speed | m s-1, km h-1 |
| Energy flux | W m-2 |

With pint installed (`pip install "climval[units]"`) other convertible units
work too.

## Options

```python
climval.compare(ref, cand, "tas")                       # units="convert" (default)
climval.compare(ref, cand, "tas", units="strict")       # equal units or UnitsError
climval.compare(ref, cand, "tas", units="ignore")       # no check, no conversion
climval.compare(ref, cand, "tas", units={"reference": "K", "candidate": "degC"})
```

The dict form declares units for inputs that lack them or carry wrong ones;
conversion then follows. A missing or unknown unit that would need a
conversion raises `UnitsError` with the two unit strings, so a silent
273.15 K bias cannot happen.

The `units` step records what was converted, for example
`candidate degC -> K`. Metric units follow: `mean_bias`, `rmse`, `mae` and
`crmse` are in the reference's units, ratios and correlations in `1`,
percentile bias in `%`.

## Temperature on an absolute scale for relative metrics

`kge` and `percentile_bias_pN` divide by a reference value, so they depend on
where zero is. When the reference is in degC or degF, `compare` computes these
two metrics with both inputs shifted to the Kelvin scale, so their values do
not depend on the temperature unit. The shift is recorded in
`result.provenance["metric_notes"]["absolute_scale"]`. Other metrics keep the
reference's units.
