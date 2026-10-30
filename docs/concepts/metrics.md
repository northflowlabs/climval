# Metrics

Every metric is computed over the points of one [sample](samples.md) with
weights \(w_i \ge 0\), which are zero where either input is missing. A weight
is the cell area, times the length of the time step in days for monthly and
yearly data when the sample pools time steps.
Differences are candidate minus reference. With the weighted mean

\[
\langle x \rangle = \frac{\sum_i w_i x_i}{\sum_i w_i},
\]

the weighted standard deviations and covariance are population values:

\[
\sigma_x = \sqrt{\langle (x - \langle x \rangle)^2 \rangle}, \qquad
\operatorname{cov}(r, c) = \langle (r - \langle r \rangle)(c - \langle c \rangle) \rangle .
\]

Every metric is an xarray reduction, so dask arrays stay lazy and one
`dask.compute` call computes all values. The test suite checks `rmse`, `mae`,
`mean_bias`, `pearson_r` and `spearman_r` against xskillscore to a relative
tolerance of 1e-10, with and without weights and missing values, and the
metrics xskillscore does not have against independent NumPy definitions to
1e-9.

| Name | Definition | Units | Best |
|---|---|---|---|
| `mean_bias` | \(\langle c - r \rangle\) | data | 0 |
| `mae` | \(\langle \lvert c - r \rvert \rangle\) | data | 0 |
| `rmse` | \(\sqrt{\langle (c - r)^2 \rangle}\) | data | 0 |
| `crmse` | \(\sqrt{\langle ((c - \langle c \rangle) - (r - \langle r \rangle))^2 \rangle}\) | data | 0 |
| `nrmse` | \(\mathrm{rmse} / \sigma_r\) | 1 | 0 |
| `pearson_r` | \(\operatorname{cov}(r, c) / (\sigma_r \sigma_c)\) | 1 | 1 |
| `spearman_r` | `pearson_r` of the ranks (by name only) | 1 | 1 |
| `std_ratio` | \(\sigma_c / \sigma_r\) | 1 | 1 |
| `taylor_skill_score` | \(\dfrac{4 (1 + R)^4}{(s + 1/s)^2 (1 + R_0)^4}\), \(R\) = `pearson_r`, \(s\) = `std_ratio`, \(R_0 = 1\) | 1 | 1 |
| `kge` | \(1 - \sqrt{(R - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}\), \(\alpha = \sigma_c / \sigma_r\), \(\beta = \langle c \rangle / \langle r \rangle\) | 1 | 1 |
| `percentile_bias_pN` | \(100\,(Q_c - Q_r) / \lvert Q_r \rvert\), \(Q\) the weighted N-th percentile | % | 0 |

`n_points` is reported with every sample: the number of points pooled, that
is points where both values are valid and the weight is positive. Points with
zero weight (for example from a zero area or time weight) take no part in any
metric and are not counted.

Metrics can be named by aliases wherever a name is taken (`metrics=`,
thresholds, `climval check` files): `bias` and `me` for `mean_bias`,
`pearson` for `pearson_r`, `spearman` for `spearman_r`, `tss` for
`taylor_skill_score`, `centred_rmse` and `centered_rmse` for `crmse`. Results
always use the canonical name. A percentile bias is named after its
percentile, with `_` for a decimal point: `percentile_bias_p95`,
`percentile_bias_p99_9` (`percentile_bias_p99.9` is read as the same metric).

## Notes on use

- **Centred RMSE and the Taylor diagram.** `crmse`, `std_ratio` and
  `pearson_r` satisfy
  \(\mathrm{crmse}^2 = \sigma_r^2 + \sigma_c^2 - 2 \sigma_r \sigma_c R\)
  (Taylor 2001, eq. 4), so a Taylor diagram can be drawn from the
  `climatology` sample alone. The total error splits as
  \(\mathrm{rmse}^2 = \mathrm{mean\_bias}^2 + \mathrm{crmse}^2\).
- **Kling-Gupta efficiency** (Gupta et al. 2009) uses a ratio of means, which
  depends on where zero is on the unit scale. It was designed for
  ratio-scale variables such as precipitation or runoff. `compare` computes
  it for temperatures in degC or degF on the Kelvin scale, so the result
  does not depend on the temperature unit.
- **Percentile bias** is relative to the reference percentile and has the
  same dependence on the zero; it is computed on the Kelvin scale in the same
  way. On samples that fit in dask's `array.chunk-size` (128 MiB, 16.8
  million float64 values) the percentile is exact; larger samples are
  estimated in the same single pass from weighted histograms with bins
  2**-20 of the value wide (`method="auto"`, the default). The estimate is
  the inverse of the weighted empirical distribution function to within one
  bin width (2**-20 of the value), clipped to the sample's minimum and
  maximum. That is not the same estimator as the exact method: for equal
  weights at most one sorted sample value lies between the two; for unequal
  weights xarray's weighted quantile interpolates differently, and the two
  can be a few sorted values apart. `PercentileBias(95, method="exact")`
  forces the exact computation (and warns when the sample exceeds
  `array.chunk-size`), and the provenance records the method each sample
  used (`metric_methods`). Points with zero weight are left out by both.
  Percentile bias is NaN where the reference percentile is below 1% of the
  reference's weighted mean absolute value (`min_relative=0.01`): the 5th
  percentile of daily precipitation, for example, is near zero, and a
  relative difference to it says nothing. The provenance then lists the
  groups under `metric_notes["percentile_bias_undefined"]`.
- **Spearman correlation** needs every point of the sample in memory at once.
  It is available by name and not computed by default. Points with zero
  weight are not ranked.
- **Constant fields.** Pearson and Spearman r, normalised RMSE, the standard
  deviation ratio, the Taylor skill score and KGE divide by a standard
  deviation (KGE also by a mean). They are NaN when an input it divides by is
  constant over the sample: when its weighted standard deviation is at most
  1e-10 of its weighted mean absolute value. The two-pass standard deviation
  of a constant field is about 1e-16 of the value, not zero, so a test for
  zero would divide by rounding noise. The NumPy interface
  (`BaseMetric.compute`) applies the same rule.
- **Errors in metrics.** An exception raised inside a metric reaches the
  caller as the same exception type (on Python 3.11 and later with a note
  naming the metric).

## Custom metrics

Subclass `climval.metrics.BaseMetric`. Implement `reduce` with the weighted
helpers to stay lazy on dask, or `compute` on NumPy arrays for a quick
prototype.

```python
from climval.metrics import BaseMetric
from climval.metrics.stats import wmean

class MeanSquaredError(BaseMetric):
    """Weighted mean squared difference."""
    name = "mse"
    higher_is_better = False
    units_kind = None

    def reduce(self, reference, candidate, weights, dims):
        return wmean((candidate - reference) ** 2, weights, dims)

result = climval.compare(ref, cand, metrics=["rmse", MeanSquaredError()])
```

## References

- Taylor, K. E. (2001). Summarizing multiple aspects of model performance in a
  single diagram. *Journal of Geophysical Research: Atmospheres*, 106(D7),
  7183-7192. <https://doi.org/10.1029/2000JD900719>
- Gleckler, P. J., Taylor, K. E., and Doutriaux, C. (2008). Performance
  metrics for climate models. *Journal of Geophysical Research: Atmospheres*,
  113(D6). <https://doi.org/10.1029/2007JD008972>
- Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F. (2009).
  Decomposition of the mean squared error and NSE performance criteria:
  Implications for improving hydrological modelling. *Journal of Hydrology*,
  377(1-2), 80-91. <https://doi.org/10.1016/j.jhydrol.2009.08.003>
