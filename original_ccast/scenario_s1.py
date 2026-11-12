#!/usr/bin/env python3
"""
Scenario S1 -- Economic re-pricing (post-hoc; NO retraining).

Takes the committed dispatch already produced by the base run and re-prices it
at an elevated value of lost load (the high end of the EU range) and across the
full real VoLL sweep, using real cost data. Reads the per-seed comparison CSVs
written by three_model_comparison.py.

Env vars:
  SAVE_DIR   directory holding comparison_seed*.csv (default results/comparison)
  C_FOSSIL   fossil price EUR/MWh (default 57, OMIE 2018 avg)
  C_CLEAN    clean variable O&M EUR/MWh (default 2, IRENA)

Run:
  SAVE_DIR=results/comparison python scenario_s1_repricing.py
"""
import os, csv, glob, statistics as st

SAVE_DIR = os.environ.get("SAVE_DIR", "results/comparison")
C_FOSSIL = float(os.environ.get("C_FOSSIL", "57"))
C_CLEAN  = float(os.environ.get("C_CLEAN", "2"))
N        = 3490
ANN      = 8760 / N
VOLL_REF = 22879.0        # Spain official
VOLL_S1  = 40000.0        # S1 stress point (above observed EU range)
SWEEP    = [4000, 10000, 22879, 40000]
MODELS   = ("ICO", "PtO", "LS")

rows = {}
for f in glob.glob(os.path.join(SAVE_DIR, "comparison_seed*.csv")):
    for r in csv.DictReader(open(f)):
        m = r["model"]
        rows.setdefault(m, []).append(
            {k: (float(v) if k not in ("model", "seed") else v) for k, v in r.items()})
if not rows:
    raise SystemExit(f"No comparison_seed*.csv found in {SAVE_DIR}")

def mean(m, k): return st.mean(r[k] for r in rows[m])

def AC(m, voll):
    gen = (C_FOSSIL * mean(m, "fossil_dispatch") + C_CLEAN * mean(m, "clean_dispatch")) * N * ANN
    cur = voll * mean(m, "total_unserved") * ANN
    return gen + cur

print(f"S1 economic re-pricing  |  {len(next(iter(rows.values())))} seeds  |  "
      f"fossil EUR{C_FOSSIL:.0f}/MWh, clean EUR{C_CLEAN:.0f}/MWh")
print("\nAnnual operating cost (EUR billion):")
print(f"{'VoLL':>9} | " + "  ".join(f"{m:>7}" for m in MODELS))
for v in SWEEP:
    print(f"{v:9,} | " + "  ".join(f"{AC(m, v)/1e9:7.1f}" for m in MODELS))

ico, pto, ls = (AC(m, VOLL_S1) for m in MODELS)
print(f"\nS1 stress point (VoLL EUR {VOLL_S1:,.0f}/MWh):")
print(f"  ICO EUR {ico/1e9:.1f} B | PtO EUR {pto/1e9:.1f} B | LS EUR {ls/1e9:.1f} B")
print(f"  ICO avoids EUR {(pto-ico)/1e9:.1f} B/yr vs PtO ({(pto-ico)/pto*100:.0f}%),"
      f" EUR {(ls-ico)/1e9:.1f} B/yr vs LS ({(ls-ico)/ls*100:.0f}%)")
