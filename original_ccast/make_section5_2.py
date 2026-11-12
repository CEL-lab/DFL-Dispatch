#!/usr/bin/env python3
# =====================================================================
#  Section 5.2 — three-model comparison (ICO vs PtO vs LS)
#  One script: main comparison TABLE + 4 comparison FIGURES.
#  Run from the project folder (Embedded_model_01), venv active.
# =====================================================================
import os, glob, csv, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CMP_GLOB = "results/comparison/comparison_seed*.csv"   # 5-seed metric CSVs
DISP_DIR = "results/comparison"                        # dispatch_{ICO,PtO,LS}_seed{SEED}.npy + y_test.npy
SEED     = "42"                                        # seed for per-hour figures (C, D)
DATA_DIR = "."
OUT      = "results/section5_2"
# ----------------------------
os.makedirs(OUT, exist_ok=True)
COL = {"ICO":"#0F6E56","PtO":"#BA7517","LS":"#993C1D"}; ORDER=["ICO","PtO","LS"]

# ---------- load 5-seed metrics ----------
rows={}
for f in sorted(glob.glob(CMP_GLOB)):
    for r in csv.DictReader(open(f)): rows.setdefault(r["model"],[]).append(r)
if not rows: sys.exit(f"No CSVs matched {CMP_GLOB}")
nseed=len(rows["ICO"])
mu=lambda m,k: float(np.mean([float(x[k]) for x in rows[m]]))
sd=lambda m,k: float(np.std ([float(x[k]) for x in rows[m]]))
def share(m,which):  # % of total dispatch
    tot=mu(m,"total_dispatch"); return mu(m,which)/tot*100 if tot else float("nan")
print(f"loaded {nseed} seeds for {list(rows)}")

# ============ TABLE 9 : full comparison (mean ± std) ============
COMP = {"min":110.0,"PtO":1.4,"LS":0.05}  # training minutes (from Sec 3.7); LS closed-form
def cell(m,k,f="{:.1f}"): return f"{mu(m,k):{f[2:-1]}} ± {sd(m,k):{f[2:-1]}}"
sec=[("Forecast accuracy",None)]
rowsT=[("Forecast accuracy","sec"),
 ("R\u00b2",      [f"{mu(m,'r2'):.4f} ± {sd(m,'r2'):.4f}" for m in ORDER]),
 ("MAE (MW)",   [f"{mu(m,'mae'):.1f} ± {sd(m,'mae'):.1f}" for m in ORDER]),
 ("RMSE (MW)",  [f"{mu(m,'rmse'):.1f} ± {sd(m,'rmse'):.1f}" for m in ORDER]),
 ("MAPE (%)",   [f"{mu(m,'mape'):.3f} ± {sd(m,'mape'):.3f}" for m in ORDER]),
 ("Bias (MW)",  [f"{mu(m,'bias'):+.1f} ± {sd(m,'bias'):.1f}" for m in ORDER]),
 ("Reliability / curtailment","sec"),
 ("Avg curtailment (MW)", [f"{mu(m,'avg_curtailment'):.1f} ± {sd(m,'avg_curtailment'):.1f}" for m in ORDER]),
 ("Curtailment rate (%)", [f"{mu(m,'curtail_rate'):.1f} ± {sd(m,'curtail_rate'):.1f}" for m in ORDER]),
 ("Total unserved (TWh)", [f"{mu(m,'total_unserved')/1e6:.3f}" for m in ORDER]),
 ("Dispatch quality","sec"),
 ("Avg QP objective", [f"{mu(m,'avg_qp_obj'):.2f}" for m in ORDER]),
 ("Operational cost (dim.)", [f"{mu(m,'avg_operational_cost'):.1f}" for m in ORDER]),
 ("Clean share (%)", [f"{share(m,'clean_dispatch'):.1f}" for m in ORDER]),
 ("Fossil share (%)", [f"{share(m,'fossil_dispatch'):.1f}" for m in ORDER]),
 ("Computational","sec"),
 ("Training time (min)", [f"~{COMP['min']:.0f}","~1.4","~0.1"]),
 ("Solver at inference", ["Yes","Yes","No"]),
]
# render
disp_rows=[[r[0]]+(["","",""] if r[1]=="sec" else r[1]) for r in rowsT]
fig,ax=plt.subplots(figsize=(9,0.46*(len(disp_rows)+2))); ax.axis("off")
t=ax.table(cellText=disp_rows,colLabels=["Metric","ICO (E2E)","PtO (two-stage)","LS (OLS+merit)"],
           cellLoc="center",loc="center",colWidths=[.34,.22,.22,.22])
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1,1.5)
secset={i+1 for i,r in enumerate(rowsT) if r[1]=="sec"}
for (rr,cc),cl in t.get_celld().items():
    cl.set_edgecolor("none")
    if rr==0: cl.set_text_props(weight="bold",color="white"); cl.set_facecolor("#0F6E56")
    elif rr in secset: cl.set_facecolor("#dcebe6"); cl.set_text_props(weight="bold",style="italic")
    else: cl.set_facecolor("#f4f7f6" if rr%2 else "white")
ax.set_title(f"Table 9.  Three-model comparison ({nseed}-seed mean ± std, 3,490 test hours)",
             fontsize=12,weight="bold",loc="left",pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/table9_comparison.png",dpi=200,bbox_inches="tight")
pd.DataFrame(disp_rows,columns=["Metric","ICO","PtO","LS"]).to_csv(f"{OUT}/table9_comparison.csv",index=False)
plt.close(fig); print("saved Table 9")

# ============ FIG 1 : accuracy vs curtailment (the core trade-off) ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
# group by metric: MAE cluster and RMSE cluster, three model bars each
gx=np.arange(2); w=0.25   # groups: MAE, RMSE
for i,m in enumerate(ORDER):
    vals=[mu(m,"mae"), mu(m,"rmse")]; errs=[sd(m,"mae"), sd(m,"rmse")]
    a1.bar(gx+(i-1)*w, vals, w, yerr=errs, capsize=3, color=COL[m], label=m, zorder=3)
a1.set_xticks(gx); a1.set_xticklabels(["MAE","RMSE"]); a1.set_ylabel("Forecast error (MW)")
a1.set_title("(a) Forecast accuracy  \u2014  ICO \u2248 PtO \u226a LS",fontsize=10.5,loc="left")
a1.legend(frameon=False,fontsize=9,loc="upper left")
a1.grid(True,axis="y",color="0.92",lw=.6)
for s in("top","right"):a1.spines[s].set_visible(False)
x=np.arange(3)
cur=[mu(m,"avg_curtailment") for m in ORDER]; cse=[sd(m,"avg_curtailment") for m in ORDER]
a2.bar(x,cur,0.6,yerr=cse,capsize=3,color=[COL[m] for m in ORDER],zorder=3)
for i,m in enumerate(ORDER): a2.text(i,cur[i]+cse[i]+4,f"{cur[i]:.0f}",ha="center",fontsize=9.5,fontweight="bold",color=COL[m])
a2.set_xticks(x); a2.set_xticklabels(ORDER); a2.set_ylabel("Avg curtailment (MW)")
a2.set_title("(b) Curtailment  \u2014  decision quality",fontsize=10.5,loc="left")
a2.grid(True,axis="y",color="0.92",lw=.6)
for s in("top","right"):a2.spines[s].set_visible(False)
fig.suptitle("ICO matches PtO accuracy while sharply reducing curtailment",fontsize=11,y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_accuracy_curtailment.png",dpi=200,bbox_inches="tight")
fig.savefig(f"{OUT}/fig_accuracy_curtailment.pdf",bbox_inches="tight"); plt.close(fig); print("saved Fig 1 (accuracy+curtailment)")

# ============ per-hour figures need dispatch matrices ============
def load_disp(m):
    p=os.path.join(DISP_DIR,f"dispatch_{m}_seed{SEED}.npy"); return np.load(p) if os.path.exists(p) else None
yp=os.path.join(DISP_DIR,"y_test.npy"); y=np.load(yp) if os.path.exists(yp) else None
Ds={m:load_disp(m) for m in ORDER}
have=all(Ds[m] is not None for m in ORDER) and y is not None

if have:
    # FIG 2 : curtailment duration curve
    fig,ax=plt.subplots(figsize=(9,4.6))
    for m in ORDER:
        sh=np.sort(np.maximum(y-Ds[m].sum(1),0))[::-1]
        ax.plot(np.arange(len(sh))/len(sh)*100, sh, color=COL[m], lw=2, label=m)
    ax.set_xlabel("Test hours ranked by shortfall (%)"); ax.set_ylabel("Hourly curtailment (MW)")
    ax.set_title("Curtailment duration curve — ICO lies below the baselines in the worst hours",fontsize=11,loc="left")
    ax.set_xlim(0,40); ax.legend(frameon=False,fontsize=10); ax.grid(True,color="0.92",lw=.6)
    for s in("top","right"):ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_curtailment_duration.png",dpi=200,bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_curtailment_duration.pdf",bbox_inches="tight"); plt.close(fig); print("saved Fig 2 (duration curve)")

    # FIG 3 : per-source dispatch comparison
    LAB=["Biomass","Brown coal","Gas","Hard coal","Oil","Hydro RoR","Hydro res.","Nuclear","Other","Other ren.","Solar","Waste","Wind"]
    o=np.argsort(-Ds["ICO"].mean(0))
    fig,ax=plt.subplots(figsize=(12,4.6)); x=np.arange(13); w=0.27
    for i,m in enumerate(ORDER):
        ax.bar(x+(i-1)*w, Ds[m].mean(0)[o], w, color=COL[m], label=m, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([LAB[j] for j in o],rotation=40,ha="right",fontsize=9)
    ax.set_ylabel("Average dispatch (MW)")
    ax.set_title("Per-source dispatch — ICO and PtO nearly identical; LS concentrates fossil",fontsize=11,loc="left")
    ax.legend(frameon=False,fontsize=10); ax.grid(True,axis="y",color="0.92",lw=.6)
    for s in("top","right"):ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_source_dispatch.png",dpi=200,bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_source_dispatch.pdf",bbox_inches="tight"); plt.close(fig); print("saved Fig 3 (per-source dispatch)")
else:
    miss=[m for m in ORDER if Ds[m] is None]+([" y_test"] if y is None else [])
    print(f"!! skipping duration curve + per-source dispatch (missing in {DISP_DIR}: {miss})")
    print(f"   need dispatch_ICO/PtO/LS_seed{SEED}.npy and y_test.npy")

print(f"\nALL OUTPUTS -> {OUT}/")
