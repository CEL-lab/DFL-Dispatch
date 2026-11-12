#!/usr/bin/env python3
# =====================================================================
#  Section 5.1 — CORRECTED Figure 3 + Table 9 block.
#
#  TWO fixes vs. your original make_section5_1.py:
#   (1) BASE-RUN ONLY.  The original globbed results/**/dispatch_ICO_seed*.npy
#       recursively, which swept in results/s2/ and results/s3/ (stress runs,
#       high fossil) and inflated Table 9's fossil share to ~18%. We now pin
#       BOTH Figure 3 and Table 9 to the base folder (results/comparison/).
#   (2) HYBRID-CAPACITY MADE EXPLICIT on Figure 3.  Dispatchable sources carry a
#       FIXED ceiling (maxtrain); variable renewables carry a ROLLING ceiling
#       (1.2*roll24max, floored at 5%). The figure now distinguishes the two
#       capacity regimes so the reader sees WHY the headroom looks different for
#       fossil vs. renewable bands (Sec. 3.4 / Eq. 6).
#
#  Drop-in: replace the "if disp_dir:" (Figure 3) and "TABLE B" blocks of your
#  make_section5_1.py with the code below. Or run this file standalone.
#
#  Run from the project folder (Embedded_model_01), venv active:
#      python make_fig3_table9_fixed.py
# =====================================================================
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
RESULTS  = "results"
DATA_DIR = "."
OUT      = "results/section5_1"
os.makedirs(OUT, exist_ok=True)

# BASE run dispatch files ONLY (no recursion, no S2/S3). Section 5.2 + CMP_GLOB
# use results/comparison/, so the base .npy files live there.
BASE_DISP_GLOB = "results/comparison/dispatch_ICO_seed*.npy"
SEED_FOR_FIG   = "42"   # Figure 3 uses one representative seed, as in the paper

TEAL, RUST = "#0F6E56", "#993C1D"
# lighter shades for the "renewable rolling ceiling" vs "fixed ceiling" hatch fills
TEAL_HDR, RUST_HDR = "#0F6E56", "#993C1D"

GEN = ["generation biomass","generation fossil brown coal/lignite","generation fossil gas",
 "generation fossil hard coal","generation fossil oil","generation hydro run-of-river and poundage",
 "generation hydro water reservoir","generation nuclear","generation other","generation other renewable",
 "generation solar","generation waste","generation wind onshore"]
LAB = ["Biomass","Brown coal","Gas","Hard coal","Oil","Hydro RoR","Hydro reservoir","Nuclear",
       "Other","Other renew.","Solar","Waste","Wind"]
FOSSIL = {1,2,3,4}                       # indices into GEN/LAB that are fossil
VRE = {"generation solar","generation wind onshore","generation hydro run-of-river and poundage",
       "generation hydro water reservoir"}
# boolean per source j: is this a variable-renewable (rolling ceiling) source?
IS_VRE = np.array([GEN[j] in VRE for j in range(13)])

# ---------------- capacity reconstruction (same logic as Sec 3.4 / your script) ----------------
def capacity_test(n_rows):
    de = pd.read_csv(os.path.join(DATA_DIR,"data","modified_dataset.csv")); de["time"]=pd.to_datetime(de["time"],utc=True)
    dw = pd.read_csv(os.path.join(DATA_DIR,"data","weather_features.csv")); dw["time"]=pd.to_datetime(dw["dt_iso"],utc=True)
    dw = dw[["time","temp","humidity","wind_speed","clouds_all"]].groupby("time").mean().reset_index()
    d = pd.merge(de,dw,on="time",how="inner").dropna(subset=["total load actual"]+GEN+["temp"])
    for lag in [1,2,3,24,168]: d[f"l{lag}"]=d["total load actual"].shift(lag)
    d["r1"]=d["total load actual"].rolling(24).mean(); d["r2"]=d["total load actual"].rolling(24).std()
    d["r3"]=d["total load actual"].rolling(168).mean(); d["r4"]=d["total load actual"].rolling(24).max()
    d["tl"]=d["temp"].shift(24); d=d.dropna().reset_index(drop=True)
    nn=len(d); tr=int(nn*0.8); vl=int(nn*0.9)
    inst=d[GEN].iloc[:tr].max().to_numpy(float)
    cap=np.zeros((nn,13))
    for j,c in enumerate(GEN):
        cap[:,j]=np.maximum(d[c].rolling(24,min_periods=1).max().to_numpy()*1.2,0.05*inst[j]) if c in VRE else inst[j]
    return cap[vl:][-n_rows:]

# ---------------- gather BASE dispatch files ----------------
base_files = sorted(glob.glob(BASE_DISP_GLOB))
if not base_files:
    raise SystemExit(f"No base dispatch files matched {BASE_DISP_GLOB} — "
                     f"edit BASE_DISP_GLOB to point at your base-run .npy folder.")

# safety: show that nothing from s2/s3 sneaks in
bad = [f for f in base_files if os.sep+"s2"+os.sep in f or os.sep+"s3"+os.sep in f]
assert not bad, f"stress-run files leaked into base list: {bad}"
print("BASE dispatch files used:", [os.path.basename(f) for f in base_files])

# pick the representative-seed file for the FIGURE
fig_file = next((f for f in base_files if f.endswith(f"seed{SEED_FOR_FIG}.npy")), base_files[0])
print("Figure 3 seed file:", os.path.basename(fig_file))

# =====================================================================
#  FIGURE 3 — per-source dispatch vs capacity, hybrid-capacity split shown
# =====================================================================
D   = np.load(fig_file)                 # (test_hours, 13)  representative seed
cap = capacity_test(len(D))             # (n_rows, 13)
dM  = D.mean(0)                         # mean dispatch per source
aM  = cap.mean(0)                       # mean available capacity per source
o   = np.argsort(-dM)                   # order bars by dispatch, high -> low

fig, ax = plt.subplots(figsize=(11.5, 4.9))
for r, j in enumerate(o):
    is_fossil = j in FOSSIL
    base_col  = RUST if is_fossil else TEAL
    # (1) solid = dispatched generation
    ax.bar(r, dM[j], color=base_col, width=.72, zorder=3)
    # (2) unused headroom, styled to make the capacity REGIME instantly distinct:
    #     rolling ceiling (variable renewable) -> SOLID pale fill   [wind, solar, hydro RoR, hydro reservoir]
    #     fixed   ceiling (dispatchable)        -> DIAGONAL hatch    [fossil, nuclear, biomass, waste, others]
    #   Solid-vs-hatched stays legible at single-column print size (Option A).
    headroom = max(aM[j] - dM[j], 0)
    if IS_VRE[j]:
        # rolling ceiling: smooth translucent block
        ax.bar(r, headroom, bottom=dM[j], facecolor=base_col, alpha=.22,
               edgecolor="none", width=.72, zorder=2)
    else:
        # fixed ceiling: open bar with diagonal hatching
        ax.bar(r, headroom, bottom=dM[j], facecolor="none", edgecolor=base_col,
               hatch="////", alpha=.9, width=.72, linewidth=0.0, zorder=2)
    # thin cap line marking the ceiling
    ax.plot([r-.36, r+.36], [aM[j], aM[j]], color=base_col, lw=1.1, alpha=.6, zorder=4)

ax.set_xticks(range(13)); ax.set_xticklabels([LAB[j] for j in o], rotation=40, ha="right", fontsize=9)
fs = dM[list(FOSSIL)].sum()/dM.sum()*100
ax.set_ylabel("Average dispatch (MW)")
ax.set_title(f"ICO per-source dispatch vs. hybrid capacity (seed {SEED_FOR_FIG}) — "
             f"clean {100-fs:.1f}% / fossil {fs:.1f}%", fontsize=12, loc="left")

# two-part legend: colour = clean/fossil ; fill style = capacity regime
legend_elems = [
    Patch(color=TEAL, label="Clean (dispatched)"),
    Patch(color=RUST, label="Fossil (dispatched)"),
    Patch(facecolor="0.5", alpha=.30, edgecolor="none",
          label="Unused — rolling ceiling (renewable)"),
    Patch(facecolor="none", edgecolor="0.3", hatch="////",
          label="Unused — fixed ceiling (dispatchable)"),
]
ax.legend(handles=legend_elems, frameon=False, fontsize=8.5, ncol=2, loc="upper right")
ax.grid(True, axis="y", color="0.92", lw=.6)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_dispatch.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_dispatch.pdf", bbox_inches="tight")
plt.close(fig)
print("saved corrected Figure 3 (hybrid-capacity split)")

# =====================================================================
#  TABLE 9 — per-source dispatch + utilisation, BASE runs only (mean +/- std)
# =====================================================================
stack, runs = [], []
for fp in base_files:
    Dr = np.load(fp)
    stack.append(Dr.mean(0)); runs.append(os.path.basename(fp))
Dm = np.vstack(stack)
disp_mu, disp_sd = Dm.mean(0), Dm.std(0)
capM = capacity_test(len(np.load(base_files[0]))).mean(0)
util = np.where(capM > 1, disp_mu/capM*100, np.nan)

# report the share so you can confirm ~91/9
tot = disp_mu.sum(); fos = disp_mu[list(FOSSIL)].sum()
print(f"\nCorrected shares (BASE only): clean {100*(tot-fos)/tot:.1f}%  |  fossil {100*fos/tot:.1f}%")
print(f"(should match Table 10: ~91.2 / 8.8)\n")

order = np.argsort(-disp_mu)
rows = []
for j in order:
    regime = "Renewable (rolling)" if IS_VRE[j] else "Dispatchable (fixed)"
    rows.append([LAB[j], "Fossil" if j in FOSSIL else "Clean", regime,
                 f"{disp_mu[j]:.0f} ± {disp_sd[j]:.0f}", f"{capM[j]:.0f}",
                 f"{util[j]:.1f}" if not np.isnan(util[j]) else "—"])
cols = ["Source","Type","Capacity regime","Avg dispatch (MW)","Avail. cap. (MW)","Util. (%)"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv(f"{OUT}/table9_decision.csv", index=False)
print("saved corrected Table 9 CSV -> results/section5_1/table9_decision.csv")
print(df.to_string(index=False))

# also render Table 9 as a styled PNG matching the paper's teal theme
fig, ax = plt.subplots(figsize=(9.2, 0.42*(len(rows)+3))); ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
             colWidths=[.17,.10,.22,.20,.16,.12])
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
for (rr,cc),cl in t.get_celld().items():
    cl.set_edgecolor("none")
    if rr == 0:
        cl.set_text_props(weight="bold", color="white"); cl.set_facecolor("#0F6E56")
    else:
        typ = rows[rr-1][1]
        cl.set_facecolor("#fbeee9" if typ == "Fossil" else ("#f4f7f6" if rr % 2 else "white"))
ax.set_title("Table 9.  ICO per-source dispatch and utilisation (base run, five-seed mean ± std)",
             fontsize=11, weight="bold", loc="left", pad=12)
fig.tight_layout()
fig.savefig(f"{OUT}/table9_decision.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved corrected Table 9 PNG")
print(f"\nBase runs aggregated: {runs}")
