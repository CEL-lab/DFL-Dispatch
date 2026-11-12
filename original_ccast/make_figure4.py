#!/usr/bin/env python
# =====================================================================
# make_figure4.py — regenerate Figure 4 (three-model comparison) from the
# fixed-run comparison CSVs. Two panels: (a) forecast error (MAE, RMSE),
# (b) average curtailment. Bar-top value labels are read from the data, so
# they always match Table 10.
#
# Usage:  RESULTS_DIR=results/comparison python make_figure4.py
# =====================================================================
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/comparison")
OUT = os.environ.get("OUT", os.path.join(RESULTS_DIR, "figure4_comparison.png"))

# ---- load five-seed means per model ----
rows = {"ICO": [], "PtO": [], "LS": []}
for f in glob.glob(os.path.join(RESULTS_DIR, "comparison_seed*.csv")):
    for r in csv.DictReader(open(f)):
        rows[r["model"]].append(r)
def mean(m, k): return np.mean([float(r[k]) for r in rows[m]])
def std(m, k):  return np.std([float(r[k]) for r in rows[m]], ddof=1)

models = ["ICO", "PtO", "LS"]
colors = ["#2ca02c", "#ff7f0e", "#8c564b"]   # ICO green, PtO orange, LS brown

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- Panel (a): forecast error (MAE and RMSE, grouped) ----
mae  = [mean(m, "mae")  for m in models]
rmse = [mean(m, "rmse") for m in models]
mae_e  = [std(m, "mae")  for m in models]
rmse_e = [std(m, "rmse") for m in models]
x = np.arange(2); w = 0.25
for i, m in enumerate(models):
    ax1.bar(x + (i-1)*w, [mae[i], rmse[i]], w, yerr=[mae_e[i], rmse_e[i]],
            capsize=3, color=colors[i], label=m)
ax1.set_xticks(x); ax1.set_xticklabels(["MAE", "RMSE"])
ax1.set_ylabel("Forecast error (MW)")
ax1.set_title("(a) Forecast accuracy   ICO $\\approx$ PtO $\\ll$ LS")
ax1.legend(frameon=False, fontsize=9)

# ---- Panel (b): average curtailment ----
cur  = [mean(m, "avg_curtailment") for m in models]
cur_e = [std(m, "avg_curtailment") for m in models]
xb = np.arange(3)
bars = ax2.bar(xb, cur, 0.6, yerr=cur_e, capsize=4, color=colors)
for i, v in enumerate(cur):
    ax2.text(i, v + cur_e[i] + 3, f"{v:.0f}", ha="center", va="bottom",
             fontweight="bold", color=colors[i])
ax2.set_xticks(xb); ax2.set_xticklabels(models)
ax2.set_ylabel("Avg curtailment (MW)")
ax2.set_title("(b) Curtailment   decision quality")
ax2.set_ylim(0, max(cur) * 1.25)

fig.suptitle("ICO matches PtO accuracy while sharply reducing curtailment", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
print("\nValues used (five-seed means):")
for m in models:
    print(f"  {m}: MAE={mean(m,'mae'):.0f}  RMSE={mean(m,'rmse'):.0f}  curtail={mean(m,'avg_curtailment'):.1f}")
