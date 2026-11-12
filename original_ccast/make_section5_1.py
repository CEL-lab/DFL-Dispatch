#!/usr/bin/env python3
# =====================================================================
#  Section 5.1 — one script: 3 images (one run) + 2 tables (aggregated)
#  Run in VS Code from the project folder (Embedded_model_01), venv active.
#  Auto-discovers history/dispatch files under results/ — no path guessing.
# =====================================================================
import os, glob, csv, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG (edit here if needed) ----------
RESULTS   = "results"        # root that holds the run folders
DATA_DIR  = "."              # folder containing data/modified_dataset.csv + weather_features.csv
SEED      = "42"             # seed used for the IMAGES (one seed)
PREFER    = "a0.8"           # try to use this run for images (your paper alpha); falls back if absent
CMP_GLOB  = "results/comparison/comparison_seed*.csv"   # 5-seed base CSVs for the forecasting table
OUT       = "results/section5_1"
# --------------------------------------------------
os.makedirs(OUT, exist_ok=True)
BLUE, RED, TEAL, RUST = "#1F4E79", "#C0392B", "#0F6E56", "#993C1D"
GEN = ["generation biomass","generation fossil brown coal/lignite","generation fossil gas",
 "generation fossil hard coal","generation fossil oil","generation hydro run-of-river and poundage",
 "generation hydro water reservoir","generation nuclear","generation other","generation other renewable",
 "generation solar","generation waste","generation wind onshore"]
LAB = ["Biomass","Brown coal","Gas","Hard coal","Oil","Hydro RoR","Hydro reservoir","Nuclear",
       "Other","Other renew.","Solar","Waste","Wind"]
FOSSIL = {1,2,3,4}
VRE = {"generation solar","generation wind onshore","generation hydro run-of-river and poundage",
       "generation hydro water reservoir"}

# ---------- discover artifacts ----------
hist = {os.path.dirname(p): p for p in glob.glob(f"{RESULTS}/**/history_ICO_seed{SEED}.npy", recursive=True)}
disp_all = sorted(glob.glob(f"{RESULTS}/**/dispatch_ICO_seed*.npy", recursive=True))
disp_seed = {os.path.dirname(p): p for p in disp_all if p.endswith(f"seed{SEED}.npy")}
print("history dirs :", list(hist) or "NONE")
print("dispatch dirs:", list({os.path.dirname(p) for p in disp_all}) or "NONE")

def pick(d):
    if not d: return None
    return next((k for k in d if PREFER in k), sorted(d)[0])
hist_dir, disp_dir = pick(hist), pick(disp_seed)
print(f"images use -> history:{hist_dir}  dispatch:{disp_dir}")
if hist_dir and disp_dir and hist_dir != disp_dir:
    print("  NOTE: loss/OC and dispatch come from DIFFERENT runs (no single run has both).")

# ---------- capacity reconstruction (for dispatch fig + table) ----------
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

# ===== IMAGES (one run) =====
if hist_dir:
    h=np.load(hist[hist_dir],allow_pickle=True).item(); ep=np.asarray(h["epoch"]); W=5
    best=int(ep[int(np.argmin(h["val_loss"]))])
    # fig1 composite loss
    f,ax=plt.subplots(figsize=(9,5.2)); ax.axvspan(ep[0],W+.5,color="0.93",lw=0)
    ax.text(ep[0],ax.get_ylim()[1],"warm-up\n(MSE only)",va="bottom",fontsize=11,color="0.4")
    ax.plot(ep,h["train_loss"],color=BLUE,lw=2.4,label="Training loss")
    ax.plot(ep,h["val_loss"],color=RED,lw=2.4,label="Validation loss")
    yb=h["val_loss"][int(np.argmin(h["val_loss"]))]; ax.axvline(best,color="0.5",ls="--",lw=1.3)
    ax.scatter([best],[yb],color=RED,s=55,zorder=5)
    ax.annotate(f"best epoch ({best})",(best,yb),textcoords="offset points",xytext=(-70,55),
                fontsize=11,color="0.25",arrowprops=dict(arrowstyle="-",color="0.6",lw=.9))
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_xlim(ep[0],ep[-1])
    ax.set_ylabel(r"Composite loss  $\alpha\,$MSE$+(1-\alpha)\,$OC$/\kappa$")
    ax.set_title(f"ICO: training vs. validation loss (seed {SEED})",fontsize=14)
    ax.legend(frameon=False,fontsize=12); ax.grid(True,which="both",color="0.92",lw=.6)
    for s in("top","right"):ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(f"{OUT}/fig1_train_val_loss.png",dpi=200,bbox_inches="tight"); f.savefig(f"{OUT}/fig1_train_val_loss.pdf",bbox_inches="tight"); plt.close(f)
    # fig2 OC
    m=ep>W; f,ax=plt.subplots(figsize=(9,5.2))
    ax.plot(ep[m],np.asarray(h["oc_train"])[m],color=BLUE,lw=2.2,label="Training OC")
    ax.plot(ep[m],np.asarray(h["oc_val"])[m],color=RED,lw=1.8,label="Validation OC")
    ax.set_xlabel("Epoch"); ax.set_xlim(ep[m][0],ep[-1]); ax.set_ylabel("Operational cost  (model units)")
    ax.set_title(f"ICO: operational cost during training (seed {SEED})",fontsize=14)
    ax.legend(frameon=False,fontsize=12); ax.grid(True,color="0.92",lw=.6)
    for s in("top","right"):ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(f"{OUT}/fig2_operational_cost.png",dpi=200,bbox_inches="tight"); f.savefig(f"{OUT}/fig2_operational_cost.pdf",bbox_inches="tight"); plt.close(f)
    print("saved fig1, fig2")
else: print("!! no history file -> skipping fig1/fig2")

if disp_dir:
    D=np.load(disp_seed[disp_dir]); cap=capacity_test(len(D)); dM=D.mean(0); aM=cap.mean(0); o=np.argsort(-dM)
    f,ax=plt.subplots(figsize=(11,4.6))
    from matplotlib.patches import Patch
    for r,j in enumerate(o):
        col=RUST if j in FOSSIL else TEAL
        ax.bar(r,dM[j],color=col,width=.7,zorder=3)
        ax.bar(r,max(aM[j]-dM[j],0),bottom=dM[j],color=col,alpha=.18,hatch="//",width=.7,zorder=2)
    ax.set_xticks(range(13)); ax.set_xticklabels([LAB[j] for j in o],rotation=40,ha="right",fontsize=9)
    fs=dM[list(FOSSIL)].sum()/dM.sum()*100; ax.set_ylabel("Average dispatch (MW)")
    ax.set_title(f"ICO per-source dispatch vs capacity (seed {SEED}) — fossil {fs:.1f}%",fontsize=12,loc="left")
    ax.legend(handles=[Patch(color=TEAL,label="Clean"),Patch(color=RUST,label="Fossil"),
              Patch(facecolor="0.7",alpha=.3,hatch="//",label="Unused capacity")],frameon=False,fontsize=9)
    ax.grid(True,axis="y",color="0.92",lw=.6)
    for s in("top","right"):ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(f"{OUT}/fig3_dispatch.png",dpi=200,bbox_inches="tight"); f.savefig(f"{OUT}/fig3_dispatch.pdf",bbox_inches="tight"); plt.close(f)
    print("saved fig3")
else: print("!! no dispatch file -> skipping fig3")

# ===== TABLE A: forecasting summary (all seeds, mean +/- std) from base CSVs =====
cmp=sorted(glob.glob(CMP_GLOB))
if cmp:
    vals={}
    for fp in cmp:
        for r in csv.DictReader(open(fp)):
            if r["model"]=="ICO":
                for k in ("r2","mae","rmse","mape","bias"): vals.setdefault(k,[]).append(float(r[k]))
    n=len(vals["mae"]); mu=lambda k:np.mean(vals[k]); sd=lambda k:np.std(vals[k])
    rowsA=[["R2",f"{mu('r2'):.4f} ± {sd('r2'):.4f}"],["MAE (MW)",f"{mu('mae'):.1f} ± {sd('mae'):.1f}"],
           ["RMSE (MW)",f"{mu('rmse'):.1f} ± {sd('rmse'):.1f}"],["MAPE (%)",f"{mu('mape'):.3f} ± {sd('mape'):.3f}"],
           ["Bias (MW)",f"{mu('bias'):+.1f} ± {sd('bias'):.1f}"]]
    pd.DataFrame(rowsA,columns=["Metric",f"ICO (mean ± std, {n} seeds)"]).to_csv(f"{OUT}/tableA_forecasting.csv",index=False)
    f,ax=plt.subplots(figsize=(7,2.6)); ax.axis("off")
    t=ax.table(cellText=rowsA,colLabels=["Metric",f"ICO  (mean ± std, {n} seeds)"],cellLoc="center",loc="center",colWidths=[.4,.6])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1,1.6)
    for (rr,cc),cl in t.get_celld().items():
        cl.set_edgecolor("none"); cl.set_facecolor("#0F6E56" if rr==0 else("#f4f7f6" if rr%2 else "white"))
        if rr==0: cl.set_text_props(weight="bold",color="white")
    ax.set_title("Table A.  ICO forecasting performance",fontsize=12,weight="bold",loc="left",pad=12)
    f.tight_layout(); f.savefig(f"{OUT}/tableA_forecasting.png",dpi=200,bbox_inches="tight"); plt.close(f)
    print(f"saved Table A (forecasting, {n} seeds)")
else: print("!! no base comparison CSVs found for Table A (set CMP_GLOB)")

# ===== TABLE B: per-source decision variables, aggregated over ALL dispatch runs =====
if disp_all:
    stack=[]; runs=[]
    for fp in disp_all:
        D=np.load(fp); cap=capacity_test(len(D))
        stack.append(D.mean(0)); runs.append(os.path.relpath(fp,RESULTS))
    Dm=np.vstack(stack); disp_mu=Dm.mean(0); disp_sd=Dm.std(0)
    capM=capacity_test(len(np.load(disp_all[0]))).mean(0)
    util=np.where(capM>1,disp_mu/capM*100,np.nan)
    rowsB=[]
    for j in np.argsort(-disp_mu):
        rowsB.append([LAB[j],"Fossil" if j in FOSSIL else "Clean",
            f"{disp_mu[j]:.0f} ± {disp_sd[j]:.0f}", f"{capM[j]:.0f}",
            f"{util[j]:.1f}" if not np.isnan(util[j]) else "—"])
    pd.DataFrame(rowsB,columns=["Source","Type","Avg dispatch (MW)","Avail cap (MW)","Util (%)"]).to_csv(f"{OUT}/tableB_decision.csv",index=False)
    f,ax=plt.subplots(figsize=(8.5,0.42*(len(rowsB)+3))); ax.axis("off")
    t=ax.table(cellText=rowsB,colLabels=["Source","Type","Avg dispatch (MW)","Avail cap (MW)","Util (%)"],
               cellLoc="center",loc="center",colWidths=[.26,.13,.25,.20,.16])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.5)
    for (rr,cc),cl in t.get_celld().items():
        cl.set_edgecolor("none")
        if rr==0: cl.set_text_props(weight="bold",color="white"); cl.set_facecolor("#0F6E56")
        else: cl.set_facecolor("#fbeee9" if rowsB[rr-1][1]=="Fossil" else ("#f4f7f6" if rr%2 else "white"))
    fsh=disp_mu[list(FOSSIL)].sum()/disp_mu.sum()*100
    ax.set_title(f"Table B.  ICO dispatch decision variables (mean ± std over {len(disp_all)} runs)",fontsize=11,weight="bold",loc="left",pad=12)
    f.text(0.02,0.005,f"Total dispatch {disp_mu.sum():.0f} MW | clean {100-fsh:.1f}% / fossil {fsh:.1f}% | runs: "+", ".join(runs),fontsize=7,style="italic",color="0.4")
    f.tight_layout(); f.savefig(f"{OUT}/tableB_decision.png",dpi=200,bbox_inches="tight"); plt.close(f)
    print(f"saved Table B (decision vars, {len(disp_all)} runs)")
else: print("!! no dispatch files for Table B")
print(f"\nALL OUTPUTS -> {OUT}/")
