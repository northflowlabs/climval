# Metrics

Every metric is computed over the points of one [sample](samples.md) with
weights \(w_i \ge 0\), which are zero where either input is missing.
Differences are candidate minus reference. With the weighted mean

\[
\langle x \rangle = \frac{\sum_i w_i x_i}{\sum_i w_i},
\]

the weighted standard deviations and covariance are population values:

\[
\sigma_x = \sqrt{\langle (x - \langle x \rangle)^2 \rangle}, \qquad
\operatorname{cov}(r, c) = \langle (r - \langle r \rangle)(c - \langle c \rangle) \rangle .
\]

Every metric is an xarray reduction, so dask arrays stay lazy and one pass
computes all values. The test suite checks the weighted metrics against
xskillscore to 1e-10, with and without weights and missing values.

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

`n_points` (the number of valid points pooled) is reported with every sample.

## Notes on use

- **Centred RMSE and the Taylor diagram.** `crmse`, `std_ratio` and
  `pearson_r` satisfy
  \(\mathrm{crmse}^2 = \sigma_r^2 + \sigma_c^2 - 2 \sigma_r \sigma_c R\)
  (Taylor 2001, eq. 4), so a Taylor diagram can be drawn from the
  `climatology` sample alone. The total error splits as
  \(\mathrm{rmse}^2 = \mathrm{mean\_bias}^2 + \mathrm{crmse}^2\).
- **Kling-Gupta efficiency** (Gupta et al. 2009) uses a ratio of means, which
  depends on where zero is on the unit scale. It is meaningful for
  ratio-scale variables such as precipitation or runoff, not for temperature
  in degC.
- **Percentile bias** (as used by Gleckler et al. 2008 for extremes) is
  relative to the reference value and has the same caveat: use K for
  temperature.
- **Spearman correlation** needs every point of the sample in memory at once.
  It is available by name and not computed by default.

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
