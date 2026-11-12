#!/usr/bin/env python3
"""
Scenario S3 -- tight capacity.  Self-contained: one file does everything.

It (1) calibrates a single capacity scale so the dispatch constraint binds in
~S3_BIND_FRAC of the test hours and scales the generation columns by it, then
(2) runs your UNMODIFIED three_model_comparison.py against the tightened data
for this SEED. You do not run anything else.

Submit one job per seed:
    for s in 42 123 456 789 1024; do qsub -v SEED=$s run_s3.pbs; done
Or run locally:
    SEED=42 BASE_DATA=. python scenario_s3.py
"""
import os, sys, shutil, subprocess
import numpy as np
import pandas as pd

SEED      = os.environ.get("SEED", "42")
BASE_DATA = os.environ.get("BASE_DATA", ".")
COMPARE   = os.environ.get("COMPARE_SCRIPT", "three_model_comparison.py")
BIND_FRAC = float(os.environ.get("S3_BIND_FRAC", "0.15"))
TARGET    = "total load actual"

GEN_COLS = [
    "generation biomass", "generation fossil brown coal/lignite",
    "generation fossil gas", "generation fossil hard coal", "generation fossil oil",
    "generation hydro run-of-river and poundage", "generation hydro water reservoir",
    "generation nuclear", "generation other", "generation other renewable",
    "generation solar", "generation waste", "generation wind onshore",
]
VARIABLE_RENEWABLE = [
    "generation solar", "generation wind onshore",
    "generation hydro run-of-river and poundage", "generation hydro water reservoir",
]

scen_dir = os.path.abspath(f"scenarios/s3/seed{SEED}")
data_dir = os.path.join(scen_dir, "data")
save_dir = os.path.abspath(f"results/s3/seed{SEED}")
os.makedirs(data_dir, exist_ok=True)

src_e = os.path.join(BASE_DATA, "data", "modified_dataset.csv")
src_w = os.path.join(BASE_DATA, "data", "weather_features.csv")

# (1a) replicate merge + feature dropna + split so the calibration aligns with the run
df_e = pd.read_csv(src_e); df_e["time"] = pd.to_datetime(df_e["time"], utc=True)
df_w = pd.read_csv(src_w); df_w["time"] = pd.to_datetime(df_w["dt_iso"], utc=True)
df_w = (df_w[["time", "temp", "humidity", "wind_speed", "clouds_all"]]
        .groupby("time").mean().reset_index())
df = pd.merge(df_e, df_w, on="time", how="inner").dropna(subset=[TARGET] + GEN_COLS + ["temp"])
for lag in [1, 2, 3, 24, 168]:
    df[f"load_lag_{lag}"] = df[TARGET].shift(lag)
df["rolling_mean_24"]  = df[TARGET].rolling(24).mean()
df["rolling_std_24"]   = df[TARGET].rolling(24).std()
df["rolling_mean_168"] = df[TARGET].rolling(168).mean()
df["rolling_max_24"]   = df[TARGET].rolling(24).max()
df["temp_lag_24"]      = df["temp"].shift(24)
df = df.dropna().reset_index(drop=True)
n = len(df); tr_end = int(n * 0.8); vl_end = int(n * 0.9)

# (1b) hybrid capacity model (Eq. 6) -> demand/availability ratio on the test split
installed = df[GEN_COLS].iloc[:tr_end].max().to_numpy(dtype=float)
avail = np.zeros((n, len(GEN_COLS)), dtype=float)
for j, col in enumerate(GEN_COLS):
    if col in VARIABLE_RENEWABLE:
        roll = df[col].rolling(24, min_periods=1).max().to_numpy() * 1.2
        avail[:, j] = np.maximum(roll, 0.05 * installed[j])
    else:
        avail[:, j] = installed[j]
r = df[TARGET].to_numpy(dtype=float)[vl_end:] / avail.sum(axis=1)[vl_end:]
scale = float(np.quantile(r, 1.0 - BIND_FRAC))
print(f"[S3] seed={SEED}  target bind={BIND_FRAC:.2f}  scale={scale:.4f}  "
      f"achieved bind={np.mean(r > scale):.3f}", flush=True)

# (1c) scale generation columns in the raw dataset and write the scenario data
df_out = pd.read_csv(src_e)
for col in GEN_COLS:
    df_out[col] = df_out[col] * scale
df_out.to_csv(os.path.join(data_dir, "modified_dataset.csv"), index=False)
shutil.copy(src_w, os.path.join(data_dir, "weather_features.csv"))

# (2) run the unmodified comparison pipeline on the tightened data
env = dict(os.environ, DATA_DIR=scen_dir, SAVE_DIR=save_dir, SEED=str(SEED))
sys.exit(subprocess.run([sys.executable, COMPARE], env=env).returncode)
