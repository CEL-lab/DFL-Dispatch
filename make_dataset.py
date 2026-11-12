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
  3. rows with a missing value in any kept column are DROPPED. The file used for the runs has
     35,061 rows (3 hours missing: 2017-11-12 20:00, 2017-11-14 11:00 and 18:00 UTC) and no NaNs;
     Kaggle's raw file has more NaNs than that, so the authors may have filled some values before
     dropping. --compare reports exactly where this script's output differs from the file used, so
     the provenance can be settled by the experiment lead rather than assumed here.
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
ap.add_argument("--fill", choices=["none", "interpolate"], default="none",
                help="how to treat NaNs before dropping (none = drop any row with a NaN)")
a = ap.parse_args()

raw = pd.read_csv(a.energy_csv)
missing = [c for c in ["time"] + GEN_COLS + [TARGET] if c not in raw.columns]
if missing: sys.exit(f"columns not found in {a.energy_csv}: {missing}")
df = raw[["time"] + GEN_COLS + [TARGET]].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
print(f"raw rows {len(df)}, NaNs per column:\n{df.isna().sum()[lambda s: s > 0].to_string() or '  none'}")
if a.fill == "interpolate":
    df = df.set_index("time")
    df[GEN_COLS + [TARGET]] = df[GEN_COLS + [TARGET]].interpolate(method="time", limit=3, limit_area="inside")
    df = df.reset_index()
n0 = len(df); df = df.dropna(subset=GEN_COLS + [TARGET]).reset_index(drop=True)
print(f"dropped {n0 - len(df)} rows with NaNs -> {len(df)} rows")
out = df.copy()
out["time"] = out["time"].dt.strftime("%-m/%-d/%Y %-H:%M")
for c in GEN_COLS + [TARGET]:
    out[c] = out[c].round().astype("int64")
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
        d = (both[c + "_ref"] - both[c + "_new"].round()).abs()
        if (d > 0).any(): print(f"  {c}: {(d > 0).sum()} values differ, max |diff| {d.max():.0f}")
    print("  value comparison done")
