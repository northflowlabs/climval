# Calendars and frequencies

Climate models use calendars that a reanalysis does not: `noleap` (365 days),
`all_leap` (366), `360_day` (twelve months of 30 days), `julian`. Their
monthly means are often stamped in the middle of the month, a reanalysis at
the start. climval therefore matches time by calendar labels, not by
timestamps.

## What happens

1. **Period.** `period=("1995", "2014")` selects on the year, month and day
   fields of each input, so it means the same months in every calendar.
   Bounds may be `"YYYY"`, `"YYYY-MM"` or `"YYYY-MM-DD"`.
2. **Native frequency.** Read from the time bounds (`time_bnds`, CF
   `bounds`) when present, then from a `frequency` attribute (CMIP
   vocabulary such as `mon`, `day`, `3hr`, `6hr`, or a pandas alias), then
   from the time steps. With bounds, each step is labelled by the midpoint of
   its interval, so values stamped at the end of their interval (ERA5
   accumulations at 00 UTC) count for the right day.
3. **Common frequency.** The coarser of the two native frequencies, or the
   one given with `freq=`: `"h"` or `"<N>h"` for a whole number of hours that
   divides 24 (`"3h"`, `"6h"`), `"D"`, `"MS"` or `"YS"`. Sub-daily data keep
   their step, so hourly against 3-hourly data is compared in 3-hour bins.
   climval aggregates but never disaggregates: a `freq` finer than an input
   raises.
4. **Aggregation.** A finer input is averaged to the common frequency.
   Monthly data aggregated to years are weighted by month length in the
   input's own calendar. A period's value counts only where at least
   `min_coverage` (default 0.8) of its native steps hold a valid value,
   counted per grid cell; periods with too few time steps are dropped and
   listed in the step record. `min_coverage=1.0` asks for complete periods.
5. **Matching.** Each period is labelled by its calendar fields (year and
   month for monthly data, year for yearly data, the date for daily data) and
   only labels present in both inputs are kept. The aligned pair carries the
   reference's time coordinate.

The `time` step in `result.provenance["steps"]` records the calendars, the
native and common frequencies, what was resampled and dropped, and how many
periods matched.

## Monthly data across calendars

Monthly and coarser comparisons are calendar independent: January 2000 of a
`360_day` model is compared with January 2000 of ERA5.

```python
result = climval.compare(era5_monthly, ukesm_tas, period=("1995", "2014"))   # 360_day model
step = next(s for s in result.provenance["steps"] if s["step"] == "time")
step["calendars"], step["matched_periods"]
```

## Daily data across calendars

Daily comparisons match days by date label, so days that exist in only one
calendar (29 February, 30 February in a `360_day` calendar, the 31st of a
month) drop out. If that matters, compare monthly means. Between a
`360_day` calendar and any other, climval warns at daily and sub-daily
frequency, because the same date label sits at a different point of the
seasonal cycle, and suggests `freq="MS"`.

## Seasons

`by="season"` groups the matched time steps into DJF, MAM, JJA and SON. By
default (`seasons="contiguous"`) a December belongs to the following year's
DJF and a season counts only when all three of its months are present (for
daily and sub-daily data, at least `min_coverage` of each month's steps), so
the partial winters at the start and end of a period are left out and listed
in `result.provenance["metric_notes"]["seasons"]`. `seasons="calendar"`
pools every December, January and February of the period instead. See
[Samples and groups](../concepts/samples.md#groups).

## Time weights

For monthly and yearly comparisons every time step is weighted by the
length of its period in days, in the reference's calendar, wherever time
steps are pooled (`space_time`, `climatology` and `time` samples). A
31-day January counts 31/28 times as much as a 28-day February, as when
months are averaged into a year. Daily and sub-daily steps count equally.

## When the frequency cannot be read

When the time axis alone cannot tell an input's frequency, climval raises
instead of guessing: a single time step says nothing about the period it
covers, steps one year apart stamped mid-year may be yearly means or one
month per year, and irregular steps have no single frequency. Pass `freq=`,
add time bounds, or declare the frequency on the input with
`da.assign_attrs(frequency="mon")`. If no period is shared, climval raises
`NoOverlapError` and names both time ranges and calendars.
