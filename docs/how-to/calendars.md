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
2. **Native frequency.** The time step of each input is classified as
   hourly, daily, monthly or yearly.
3. **Common frequency.** The coarser of the two native frequencies, or the
   one given with `freq=` (`"h"`, `"D"`, `"MS"`, `"YS"`). climval aggregates
   but never disaggregates: a `freq` finer than an input raises.
4. **Aggregation.** A finer input is averaged to the common frequency.
   Monthly data aggregated to years are weighted by month length in the
   input's own calendar. Periods with fewer time steps than they should hold
   are dropped and listed in the step record.
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
month) drop out. If that matters, compare monthly means.

## Seasons

`by="season"` groups the matched time steps into DJF, MAM, JJA and SON by
their month. With monthly data every season of every year contributes its
three months when they are in the period.

## When the time step is irregular

When the native step cannot be classified (for example a subset of months),
pass `freq=` explicitly. If no period is shared, climval raises
`NoOverlapError` and names both time ranges and calendars.
