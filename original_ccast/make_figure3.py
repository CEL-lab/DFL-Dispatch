#!/usr/bin/env python
# =====================================================================
# make_figure3.py — regenerate Figure 3 (ICO average per-source dispatch vs
# available capacity) from the fixed-run dispatch arrays + the dataset.
# Reconstructs the hourly capacity exactly as the training script does
# (verified: dispatch never exceeds it), then plots dispatched MW with the
# unused headroom stacked on top, split by clean/fossil and rolling/fixed.
#
# Usage:
#   RESULTS_DIR=results/ico_model DATA=modified_dataset.csv python make_figure3.py
# =====================================================================
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/ico_model")
DATA = os.environ.get("DATA", "modified_dataset.csv")
OUT  = os.environ.get("OUT", os.path.join(RESULTS_DIR, "figure3_dispatch.png"))
HEADROOM, FLOOR = 1.2, 0.05

GEN_COLS = ["generation biomass","generation fossil brown coal/lignite","generation fossil gas",
 "generation fossil hard coal","generation fossil oil","generation hydro run-of-river and poundage",
 "generation hydro water reservoir","generation nuclear","generation other","generation other renewable",
 "generation solar","generation waste","generation wind onshore"]
VARIABLE_RENEWABLE = {"generation solar","generation wind onshore",
 "generation hydro run-of-river and poundage","generation hydro water reservoir"}
FOSSIL = {"generation fossil brown coal/lignite","generation fossil gas",
          "generation fossil hard coal","generation fossil oil"}
PRETTY = {"generation wind onshore":"Wind","generation nuclear":"Nuclear",
 "generation hydro water reservoir":"Hydro reservoir","generation solar":"Solar",
 "generation hydro run-of-river and poundage":"Hydro RoR","generation fossil hard coal":"Hard coal",
 "generation fossil gas":"Gas","generation biomass":"Biomass","generation fossil brown coal/lignite":"Brown coal",
 "generation waste":"Waste","generation fossil oil":"Oil","generation other renewable":"Other renew.","generation other":"Other"}

# ---- rebuild the exact test-window hourly capacity ----
d = pd.read_csv(DATA)
for k in [1,2,3,24,168]: d[f"l{k}"] = d["total load actual"].shift(k)
d["a"]=d["total load actual"].rolling(24).mean(); d["b"]=d["total load actual"].rolling(24).std()
d["c"]=d["total load actual"].rolling(168).mean(); d["e"]=d["total load actual"].rolling(24).max()
d = d.dropna().reset_index(drop=True); n = len(d)
tr_end, vl_end = int(n*0.8), int(n*0.9)
installed = d[GEN_COLS].iloc[:tr_end].max().values.astype(float)
caps = np.zeros((n, 13))
for j, c in enumerate(GEN_COLS):
    caps[:, j] = (np.maximum(d[c].rolling(24, min_periods=1).max().values*HEADROOM, FLOOR*installed[j])
                  if c in VARIABLE_RENEWABLE else installed[j])
cap_test = caps[vl_end:n]
avail = cap_test.mean(axis=0)                     # mean hourly ceiling

# ---- average dispatch across seeds ----
disp = np.stack([np.load(f) for f in sorted(glob.glob(os.path.join(RESULTS_DIR,"dispatch_ICO_seed*.npy")))], 0)
avg = disp.mean(axis=(0,1))

# verify (sanity): dispatch <= ceiling
assert (disp.mean(0) <= cap_test + 1.0).all(), "dispatch exceeds reconstructed ceiling — check GEN_COLS order"

# ---- order by dispatch, plot dispatched + unused headroom ----
order = np.argsort(-avg)
labels = [PRETTY[GEN_COLS[j]] for j in order]
disp_o = avg[order]; avail_o = avail[order]; unused_o = np.maximum(avail_o - disp_o, 0)
is_foss = np.array([GEN_COLS[j] in FOSSIL for j in order])
is_var  = np.array([GEN_COLS[j] in VARIABLE_RENEWABLE for j in order])

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(order))
# dispatched (solid): green clean, brown fossil
for i in range(len(order)):
    ax.bar(x[i], disp_o[i], color=("#8c564b" if is_foss[i] else "#2ca02c"), zorder=3)
# unused headroom (hatched/light) on top
for i in range(len(order)):
    ax.bar(x[i], unused_o[i], bottom=disp_o[i],
           color="none", edgecolor=("#8c564b" if is_foss[i] else "#7fbf7f"),
           hatch="//", linewidth=0.6, zorder=2)

ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Average dispatch (MW)")
ax.set_title("ICO average per-source dispatch vs available capacity")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#2ca02c", label="Clean (dispatched)"),
    Patch(color="#8c564b", label="Fossil (dispatched)"),
    Patch(facecolor="none", edgecolor="#7fbf7f", hatch="//", label="Unused headroom"),
], frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
print("\nPer-source (avg dispatch / mean avail / util%):")
for j in order:
    u = 100*avg[j]/max(avail[GEN_COLS.index(GEN_COLS[j])],1e-9)
    print(f"  {PRETTY[GEN_COLS[j]]:<16} {avg[j]:7.0f} / {avail[order][list(order).index(j)]:7.0f}")
