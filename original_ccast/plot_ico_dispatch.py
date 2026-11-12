#!/usr/bin/env python3
"""
Figure 3 — ICO per-source generation dispatch vs available capacity.

Reads the per-source dispatch matrix three_model_comparison.py already saves
(dispatch_ICO_seed{SEED}.npy, shape N x 13) and reconstructs the hybrid capacity
ceiling from the dataset (no re-run needed). Produces a bar chart of average
dispatch per source with unused capacity headroom hatched, fossil vs clean
colored.

Usage:
  DATA_DIR=. SAVE_DIR=results/comparison SEED=42 python plot_ico_dispatch.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", ".")
SAVE_DIR = os.environ.get("SAVE_DIR", "results/comparison")
SEED     = os.environ.get("SEED", "42")
TARGET   = "total load actual"

GEN_COLS = ["generation biomass","generation fossil brown coal/lignite","generation fossil gas",
    "generation fossil hard coal","generation fossil oil","generation hydro run-of-river and poundage",
    "generation hydro water reservoir","generation nuclear","generation other","generation other renewable",
    "generation solar","generation waste","generation wind onshore"]
LABEL = ["Biomass","Brown coal","Gas","Hard coal","Oil","Hydro RoR","Hydro reservoir","Nuclear",
         "Other","Other renew.","Solar","Waste","Wind"]
FOSSIL = {1,2,3,4}                                   # brown coal, gas, hard coal, oil
VARIABLE_RENEWABLE = {"generation solar","generation wind onshore",
    "generation hydro run-of-river and poundage","generation hydro water reservoir"}

# ---- load saved dispatch ----
D = np.load(os.path.join(SAVE_DIR, f"dispatch_ICO_seed{SEED}.npy"))   # N x 13 (MW)

# ---- reconstruct the per-source capacity ceiling (Eq. 6), aligned to the test split ----
df_e = pd.read_csv(os.path.join(DATA_DIR,"data","modified_dataset.csv")); df_e["time"]=pd.to_datetime(df_e["time"],utc=True)
df_w = pd.read_csv(os.path.join(DATA_DIR,"data","weather_features.csv")); df_w["time"]=pd.to_datetime(df_w["dt_iso"],utc=True)
df_w = df_w[["time","temp","humidity","wind_speed","clouds_all"]].groupby("time").mean().reset_index()
df = pd.merge(df_e,df_w,on="time",how="inner").dropna(subset=[TARGET]+GEN_COLS+["temp"])
for lag in [1,2,3,24,168]: df[f"load_lag_{lag}"]=df[TARGET].shift(lag)
df["rolling_mean_24"]=df[TARGET].rolling(24).mean(); df["rolling_std_24"]=df[TARGET].rolling(24).std()
df["rolling_mean_168"]=df[TARGET].rolling(168).mean(); df["rolling_max_24"]=df[TARGET].rolling(24).max()
df["temp_lag_24"]=df["temp"].shift(24); df=df.dropna().reset_index(drop=True)
n=len(df); tr_end=int(n*0.8); vl_end=int(n*0.9)
installed=df[GEN_COLS].iloc[:tr_end].max().to_numpy(float)
cap=np.zeros((n,len(GEN_COLS)))
for j,c in enumerate(GEN_COLS):
    if c in VARIABLE_RENEWABLE:
        cap[:,j]=np.maximum(df[c].rolling(24,min_periods=1).max().to_numpy()*1.2, 0.05*installed[j])
    else:
        cap[:,j]=installed[j]
cap_test = cap[vl_end:][-len(D):]                    # align to the N test rows

disp = D.mean(0); avail = cap_test.mean(0)
order = np.argsort(-disp)                            # largest dispatch first

fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(GEN_COLS))
for rank, j in enumerate(order):
    col = "#993C1D" if j in FOSSIL else "#0F6E56"
    ax.bar(rank, disp[j], color=col, width=0.7, zorder=3)
    head = max(avail[j]-disp[j], 0)
    ax.bar(rank, head, bottom=disp[j], color=col, width=0.7, alpha=0.18, hatch="//",
           edgecolor=col, lw=0, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([LABEL[j] for j in order], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Average dispatch (MW)")
fossil_sh = disp[list(FOSSIL)].sum()/disp.sum()*100
ax.set_title(f"ICO per-source dispatch vs available capacity (seed {SEED})  \u2014  "
             f"fossil share {fossil_sh:.1f}%", fontsize=11, loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#0F6E56",label="Clean (dispatched)"),
                   Patch(color="#993C1D",label="Fossil (dispatched)"),
                   Patch(facecolor="0.7",alpha=0.3,hatch="//",label="Unused capacity")],
          frameon=False, fontsize=9)
ax.grid(True, axis="y", color="0.92", lw=0.6)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
out=os.path.join(SAVE_DIR,f"fig_ico_dispatch_seed{SEED}")
fig.savefig(out+".png",dpi=200,bbox_inches="tight"); fig.savefig(out+".pdf",bbox_inches="tight")
print("saved",out+".png / .pdf"); print(f"fossil share {fossil_sh:.1f}%  total dispatch {disp.sum():.0f} MW")
