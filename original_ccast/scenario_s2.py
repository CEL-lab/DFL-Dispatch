#!/usr/bin/env python3
"""
Scenario S2 -- demand volatility.  Self-contained: one file does everything.

It (1) adds Gaussian noise (sigma = NOISE_SIGMA MW) to the realised load, then
(2) runs your UNMODIFIED three_model_comparison.py against the noisy data for
this SEED. You do not run anything else.

Submit one job per seed:
    for s in 42 123 456 789 1024; do qsub -v SEED=$s run_s2.pbs; done
Or run locally:
    SEED=42 BASE_DATA=. python scenario_s2.py
"""
import os, sys, shutil, subprocess
import numpy as np
import pandas as pd

SEED        = os.environ.get("SEED", "42")
BASE_DATA   = os.environ.get("BASE_DATA", ".")               # holds data/modified_dataset.csv + weather_features.csv
COMPARE     = os.environ.get("COMPARE_SCRIPT", "three_model_comparison.py")
NOISE_SIGMA = float(os.environ.get("NOISE_SIGMA", "2000"))
NOISE_SEED  = int(os.environ.get("NOISE_SEED", "20240101"))  # fixed -> every model seed sees the same noisy data
TARGET      = "total load actual"

scen_dir = os.path.abspath(f"scenarios/s2/seed{SEED}")
data_dir = os.path.join(scen_dir, "data")
save_dir = os.path.abspath(f"results/s2/seed{SEED}")
os.makedirs(data_dir, exist_ok=True)

# (1) apply the scenario condition: noise on the load column
src_e = os.path.join(BASE_DATA, "data", "modified_dataset.csv")
src_w = os.path.join(BASE_DATA, "data", "weather_features.csv")
df = pd.read_csv(src_e)
clean = df[TARGET].to_numpy(dtype=float)
noise = np.random.default_rng(NOISE_SEED).normal(0.0, NOISE_SIGMA, size=len(df))
df[TARGET] = np.maximum(clean + noise, 1.0)
df.to_csv(os.path.join(data_dir, "modified_dataset.csv"), index=False)
shutil.copy(src_w, os.path.join(data_dir, "weather_features.csv"))
print(f"[S2] seed={SEED}  sigma={NOISE_SIGMA:.0f} MW  "
      f"mean|load shift|={np.mean(np.abs(df[TARGET].to_numpy()-clean)):.0f} MW", flush=True)

# (2) run the unmodified comparison pipeline on the noisy data
env = dict(os.environ, DATA_DIR=scen_dir, SAVE_DIR=save_dir, SEED=str(SEED))
sys.exit(subprocess.run([sys.executable, COMPARE], env=env).returncode)
