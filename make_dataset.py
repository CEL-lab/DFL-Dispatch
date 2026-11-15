#!/usr/bin/env python
"""Build modified_dataset.csv from the Kaggle "Hourly energy demand generation and weather" files.

Source (Jhana, 2019): https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
  energy_dataset.csv   (35,064 hourly rows, 2015-01-01 00:00+01:00 .. 2018-12-31 23:00+01:00)
  weather_features.csv (used unchanged by run_experiment.py)

Usage:  python make_dataset.py <path/to/energy_dataset.csv> [--out modified_dataset.csv] [--compare existing.csv]

Transformation (documented here because no script for it existed before 1 Oct 2026):
  1. keep the 13 non-zero generation columns used in the paper, the pumped-storage consumption
     column (carried but unused), and "total load actual";
  2. convert the tz-aware "time" column to UTC and write it as "M/D/YYYY H:MM" (the format of the
     file used for all runs; e.g. 2015-01-01 00:00+01:00 -> 12/31/2014 23:00);
  3. missing values (rule reconstructed from the raw Kaggle file, Nov 2026; reproduces the file
     used for all runs exactly). The raw file has 46 hours with a missing value in the kept
     columns. In 44 of them every missing value -- total load in 34 hours, generation in 23 --
     was filled with the mean of the same UTC weekday and hour over the full raw sample. Two
     hours with a missing load value (2017-11-14 11:00 and 18:00 UTC) were dropped instead, and
     one complete hour (2017-11-12 20:00 UTC) was also dropped; the reason for treating these
     three hours differently is not documented. Result: 35,061 rows, no missing values.
     Imputed hours by year: 32 in 2015, 10 in 2016, 2 in 2018 (June and July). None is in the
     base test period (from 2018-08-09); the rolling-origin test years 2016 and 2018 contain 10
     and 2 imputed hours.
"""
import argparse, sys
import numpy as np
import pandas as pd

GEN_COLS = [
    "generation biomass", "generation fossil brown coal/lignite", "generation fossil gas",
    "generation fossil hard coal", "generation fossil oil", "generation hydro pumped storage consumption",
    "generation hydro run-of-river and poundage", "generation hydro water reservoir",
    "generation nuclear", "generation other", "generation other renewable",
    "generation solar", "generation waste", "generation wind onshore"]
TARGET = "total load actual"

ap = argparse.ArgumentParser()
ap.add_argument("energy_csv")
ap.add_argument("--out", default="modified_dataset.csv")
ap.add_argument("--compare", default=None, help="existing modified_dataset.csv to diff against")
ap.add_argument("--fill", choices=["dowhour", "none", "interpolate"], default="dowhour",
                help="dowhour = rule used for the paper's data (fill every missing value with the full-sample "
                     "mean for the same UTC weekday and hour; drop three listed hours); "
                     "none = drop any row with a NaN; interpolate = time interpolation, then drop")
DROPPED_HOURS = ["2017-11-12 20:00:00+00:00", "2017-11-14 11:00:00+00:00", "2017-11-14 18:00:00+00:00"]
a = ap.parse_args()

raw = pd.read_csv(a.energy_csv)
missing = [c for c in ["time"] + GEN_COLS + [TARGET] if c not in raw.columns]
if missing: sys.exit(f"columns not found in {a.energy_csv}: {missing}")
df = raw[["time"] + GEN_COLS + [TARGET]].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
print(f"raw rows {len(df)}, NaNs per column:\n{df.isna().sum()[lambda s: s > 0].to_string() or '  none'}")
if a.fill == "dowhour":
    key = [df["time"].dt.dayofweek, df["time"].dt.hour]                 # UTC weekday and hour
    means = df.groupby(key)[GEN_COLS + [TARGET]].transform("mean")      # over the full raw sample
    drop = df["time"].isin(pd.to_datetime(DROPPED_HOURS, utc=True))
    filled = df[GEN_COLS + [TARGET]].isna().any(axis=1) & ~drop
    print(f"filling {int(filled.sum())} hours (load missing in {int(df.loc[~drop, TARGET].isna().sum())}) "
          f"with the weekday-hour mean")
    df[GEN_COLS + [TARGET]] = df[GEN_COLS + [TARGET]].fillna(means)
    df = df[~drop].reset_index(drop=True)
    print(f"dropped {int(drop.sum())} hours as in the file used: {DROPPED_HOURS}")
elif a.fill == "interpolate":
    df = df.set_index("time")
    df[GEN_COLS + [TARGET]] = df[GEN_COLS + [TARGET]].interpolate(method="time", limit=3, limit_area="inside")
    df = df.reset_index()
n0 = len(df); df = df.dropna(subset=GEN_COLS + [TARGET]).reset_index(drop=True)
print(f"dropped {n0 - len(df)} rows with NaNs -> {len(df)} rows")
out = df.copy()
out["time"] = out["time"].dt.strftime("%-m/%-d/%Y %-H:%M")
for c in GEN_COLS + [TARGET]:          # raw values are whole MW; filled means keep their decimals
    if (out[c] % 1 == 0).all():
        out[c] = out[c].astype("int64")
out.to_csv(a.out, index=False)
print(f"wrote {a.out}")

if a.compare:
    ref = pd.read_csv(a.compare); ref["time"] = pd.to_datetime(ref["time"], utc=True)
    new = df.copy()
    m = pd.merge(ref, new, on="time", how="outer", suffixes=("_ref", "_new"), indicator=True)
    only_ref = m[m["_merge"] == "left_only"]["time"]; only_new = m[m["_merge"] == "right_only"]["time"]
    print(f"\ncompare with {a.compare}: rows ref {len(ref)}, new {len(new)}; "
          f"hours only in ref {len(only_ref)}, only in new {len(only_new)}")
    if len(only_ref): print("  only in ref (first 10):", only_ref.head(10).astype(str).tolist())
    if len(only_new): print("  only in new (first 10):", only_new.head(10).astype(str).tolist())
    both = m[m["_merge"] == "both"]
    for c in GEN_COLS + [TARGET]:
        d = (both[c + "_ref"] - both[c + "_new"]).abs()
        if (d > 1e-4).any(): print(f"  {c}: {(d > 1e-4).sum()} values differ, max |diff| {d.max():.4f}")
    print("  value comparison done")
