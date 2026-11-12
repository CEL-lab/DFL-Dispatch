#!/usr/bin/env python3
# =====================================================================
#  Section 5.3 — Robustness, Economic, and Sensitivity Analysis
#  Generates all figures + tables for 5.3.1 (scenarios), 5.3.2 (economics),
#  5.3.3 (alpha ablation). Reads your base / S2 / S3 / alpha CSVs.
#  Run from the project folder (Embedded_model_01), venv active.
# =====================================================================
import os, glob, csv, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- CONFIG ----------
BASE_GLOB  = "results/comparison/comparison_seed*.csv"
S2_GLOB    = "results/s2/**/comparison_seed*.csv"
S3_GLOB    = "results/s3/**/comparison_seed*.csv"
ALPHA_ROOT = "results/alpha"            # holds a0.0/, a0.5/, a0.9/ ... /comparison_seed42.csv
SEED       = "42"                       # seed for the alpha frontier
# economic valuation prices
P_FOS, P_CLEAN, VOLL = 57.0, 2.0, 22879.0
VOLL_GRID = [4000,8000,12000,16000,22879,30000,40000,69000]
N_TEST, ANN = 3490, 8760/3490
OUT = "results/section5_3"
# ----------------------------
os.makedirs(OUT, exist_ok=True)
COL={"ICO":"#0F6E56","PtO":"#BA7517","LS":"#993C1D"}; ORDER=["ICO","PtO","LS"]
GREENS=["#3A7D6B","#0F6E56","#0B5240"]

def load(globpat):
    rows={}
    for f in sorted(glob.glob(globpat, recursive=True)):
        for r in csv.DictReader(open(f)): rows.setdefault(r["model"],[]).append(r)
    return rows
def M(rows,m,k): return float(np.mean([float(x[k]) for x in rows[m]])) if rows.get(m) else float("nan")
def S(rows,m,k): return float(np.std ([float(x[k]) for x in rows[m]])) if rows.get(m) else float("nan")

base=load(BASE_GLOB); s2=load(S2_GLOB); s3=load(S3_GLOB)
print("base seeds:", len(base.get("ICO",[])), "| S2:", len(s2.get("ICO",[])), "| S3:", len(s3.get("ICO",[])))

# ---------- helper: 3-panel scenario figure ----------
def scenario_fig(rows, title, fname):
    if not rows.get("ICO"): print(f"!! no data for {fname}"); return
    rng=np.random.default_rng(3)
    fig,ax=plt.subplots(1,3,figsize=(12.5,4.2))
    def panel(a,key,ttl,ylab,zero=False,fmt="{:.0f}"):
        means=[M(rows,m,key) for m in ORDER]
        allv=[[float(x[key]) for x in rows[m]] for m in ORDER]
        span=max(max(v) for v in allv)-min(min(v) for v in allv) or 1
        a.bar(range(3),means,color=[COL[m] for m in ORDER],width=0.62,edgecolor="white",zorder=2)
        for i,v in enumerate(allv):
            a.scatter(i+(rng.random(len(v))-.5)*0.18,v,s=20,color="white",edgecolor="0.25",lw=.7,zorder=3)
        for i,m in enumerate(ORDER):
            a.text(i,max(allv[i])+span*0.06,fmt.format(means[i]),ha="center",va="bottom",
                   fontsize=9.5,color=COL[m],fontweight="bold")
        if zero: a.axhline(0,color="0.4",lw=.8)
        a.set_xticks(range(3)); a.set_xticklabels(ORDER); a.set_title(ttl,fontsize=10.5,loc="left")
        a.set_ylabel(ylab); a.grid(True,axis="y",color="0.9",lw=.6)
        for s in("top","right"): a.spines[s].set_visible(False)
        lo=min(0,min(min(v) for v in allv)); hi=max(max(v) for v in allv)
        a.set_ylim(lo-span*0.05,hi+span*0.18)
    rd=(M(rows,"PtO","avg_curtailment")-M(rows,"ICO","avg_curtailment"))/M(rows,"PtO","avg_curtailment")*100
    panel(ax[0],"avg_curtailment",f"(a) Curtailment \u2014 ICO {rd:.0f}% below PtO","Avg. curtailment (MW)")
    panel(ax[1],"bias","(b) Safety margin (forecast bias)","Bias = forecast \u2212 actual (MW)",zero=True,fmt="{:+.0f}")
    panel(ax[2],"avg_operational_cost","(c) Operational cost (dim.)","Avg. operational cost")
    fig.suptitle(title,fontsize=11,y=1.02); fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{OUT}/{fname}.pdf",bbox_inches="tight")
    plt.close(fig); print("saved",fname)

scenario_fig(s2,"Scenario S2 \u2014 demand volatility (\u03c3=2000 MW), 5 seeds","s2_results")
scenario_fig(s3,"Scenario S3 \u2014 tight capacity (binds 15% of hours), 5 seeds","s3_results")

# ---------- cross-scenario summary figure + table ----------
def curt(r,m): return M(r,m,"avg_curtailment")
def bias(r,m): return M(r,m,"bias")
conds=[("Base",base),("S2\nvolatility",s2),("S3\ntight cap.",s3)]
have_all=all(r.get("ICO") for _,r in conds)
if have_all:
    redux=[(curt(r,"PtO")-curt(r,"ICO"))/curt(r,"PtO")*100 for _,r in conds]
    margin=[bias(r,"ICO") for _,r in conds]
    fig,(b1,b2)=plt.subplots(1,2,figsize=(9.5,4.0)); labels=[c for c,_ in conds]
    b1.bar(labels,redux,color=GREENS,width=.6)
    for i,v in enumerate(redux): b1.text(i,v+0.6,f"{v:.0f}%",ha="center",fontsize=10,fontweight="bold",color="#0B5240")
    b1.set_ylabel("ICO curtailment reduction vs PtO (%)"); b1.set_title("(a) Decision-focused advantage",fontsize=10.5,loc="left")
    b1.set_ylim(0,max(redux)*1.25); b1.grid(True,axis="y",color="0.9",lw=.6)
    for s in("top","right"): b1.spines[s].set_visible(False)
    b2.bar(labels,margin,color=GREENS,width=.6)
    for i,v in enumerate(margin): b2.text(i,v+max(margin)*0.02,f"+{v:.0f}",ha="center",fontsize=10,fontweight="bold",color="#0B5240")
    b2.set_ylabel("ICO safety margin (bias, MW)"); b2.set_title("(b) Safety margin adapts to the stressor",fontsize=10.5,loc="left")
    b2.set_ylim(0,max(margin)*1.2); b2.grid(True,axis="y",color="0.9",lw=.6)
    for s in("top","right"): b2.spines[s].set_visible(False)
    fig.suptitle("Cross-scenario summary \u2014 ICO vs PtO across operating conditions (5 seeds)",fontsize=11,y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/cross_scenario_summary.png",dpi=200,bbox_inches="tight")
    fig.savefig(f"{OUT}/cross_scenario_summary.pdf",bbox_inches="tight"); plt.close(fig); print("saved cross_scenario_summary")

    # cross-scenario TABLE (image)
    rowsX=[]
    for (c,r),name in zip(conds,["Base","S2 (volatility)","S3 (tight cap.)"]):
        rd=(curt(r,"PtO")-curt(r,"ICO"))/curt(r,"PtO")*100
        rowsX.append([name,f"{curt(r,'ICO'):.1f}",f"{curt(r,'PtO'):.1f}",f"\u2212{rd:.0f}%",
                      f"+{bias(r,'ICO'):.0f}",f"{bias(r,'PtO'):+.0f}"])
    _table(rowsX,["Condition","ICO curt.","PtO curt.","Reduction","ICO bias","PtO bias"],
           "Table 10.  Decision quality across operating conditions (5-seed means)","table10_cross_scenario",
           [.24,.15,.15,.16,.15,.15]) if False else None
    # (table rendered below via _table which is defined next)
    _cross_rows=rowsX
else:
    print("!! cross-scenario needs base + S2 + S3"); _cross_rows=None

# ---------- generic table renderer ----------
def _table(rows,cols,title,fname,widths,secrows=None,boldcol=None):
    fig,ax=plt.subplots(figsize=(1.6*len(cols),0.5*(len(rows)+2))); ax.axis("off")
    t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center",colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.5)
    for (r,c),cl in t.get_celld().items():
        cl.set_edgecolor("none")
        if r==0: cl.set_text_props(weight="bold",color="white"); cl.set_facecolor("#0F6E56")
        elif secrows and r in secrows: cl.set_facecolor("#dcebe6"); cl.set_text_props(weight="bold",style="italic")
        else: cl.set_facecolor("#f4f7f6" if r%2 else "white")
    ax.set_title(title,fontsize=11,weight="bold",loc="left",pad=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows,columns=cols).to_csv(f"{OUT}/{fname}.csv",index=False); print("saved",fname)

if _cross_rows:
    _table(_cross_rows,["Condition","ICO curt. (MW)","PtO curt. (MW)","Reduction","ICO bias (MW)","PtO bias (MW)"],
           "Table 10.  Decision quality across operating conditions (5-seed means)","table10_cross_scenario",
           [.22,.16,.16,.14,.16,.16])

# ---------- 5.3.2 economic: VoLL sweep figure + AC table ----------
if base.get("ICO"):
    def AC(m,v): return (P_FOS*M(base,m,"fossil_dispatch")+P_CLEAN*M(base,m,"clean_dispatch"))*N_TEST*ANN + v*M(base,m,"total_unserved")*ANN
    # figure
    fig,(axL,axR)=plt.subplots(1,2,figsize=(11.5,4.4),width_ratios=[1.3,1])
    vg=np.linspace(0,69000,300)
    for m in ORDER: axL.plot(vg/1000,[AC(m,v)/1e9 for v in vg],color=COL[m],lw=2.2,label=m)
    axL.axvspan(4,69,color="0.9",lw=0,zorder=0); axL.axvline(22.879,color="0.45",ls="--",lw=1)
    axL.text(23.4,axL.get_ylim()[1]*0.1,"Spain VoLL\n\u20ac22,879",fontsize=8.5,color="0.4")
    axL.set_xlabel("Value of lost load (\u20ac thousand / MWh)"); axL.set_ylabel("Annual operating cost (\u20ac billion)")
    axL.set_title("(a) Cost vs reliability price",fontsize=10.5,loc="left"); axL.legend(frameon=False,fontsize=10)
    axL.grid(True,color="0.92",lw=.6)
    for s in("top","right"): axL.spines[s].set_visible(False)
    gens={m:(P_FOS*M(base,m,"fossil_dispatch")+P_CLEAN*M(base,m,"clean_dispatch"))*N_TEST*ANN/1e9 for m in ORDER}
    curts={m:VOLL*M(base,m,"total_unserved")*ANN/1e9 for m in ORDER}
    axR.bar(ORDER,[gens[m] for m in ORDER],color="#9FE1CB",label="Generation")
    axR.bar(ORDER,[curts[m] for m in ORDER],bottom=[gens[m] for m in ORDER],color=[COL[m] for m in ORDER],label="Curtailment (VoLL)")
    for i,m in enumerate(ORDER): axR.text(i,gens[m]+curts[m]+1,f"{gens[m]+curts[m]:.1f}",ha="center",fontsize=9.5,fontweight="bold")
    axR.set_ylabel("Annual cost (\u20ac billion)"); axR.set_title("(b) At Spain's official VoLL",fontsize=10.5,loc="left")
    axR.legend(frameon=False,fontsize=9)
    for s in("top","right"): axR.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/voll_sweep.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{OUT}/voll_sweep.pdf",bbox_inches="tight")
    plt.close(fig); print("saved voll_sweep")
    # AC table
    rowsV=[]
    for v in VOLL_GRID:
        i,p,l=AC("ICO",v)/1e9,AC("PtO",v)/1e9,AC("LS",v)/1e9
        rowsV.append([f"{v:,}"+("  (Spain)" if v==22879 else ""),f"{i:.1f}",f"{p:.1f}",f"{l:.1f}",f"{p-i:.1f}",f"{l-i:.1f}"])
    _table(rowsV,["VoLL (\u20ac/MWh)","ICO","PtO","LS","ICO vs PtO","ICO vs LS"],
           "Table 11.  Annual operating cost (\u20ac B/yr) and ICO savings across the VoLL range","table11_voll",
           [.24,.13,.13,.13,.18,.18])
    be=((P_FOS*(M(base,"ICO","fossil_dispatch")-M(base,"PtO","fossil_dispatch"))+P_CLEAN*(M(base,"ICO","clean_dispatch")-M(base,"PtO","clean_dispatch")))*N_TEST)/(M(base,"PtO","total_unserved")-M(base,"ICO","total_unserved"))
    print(f"   break-even VoLL (ICO vs PtO): EUR {be:.0f}/MWh")

# ---------- 5.3.3 alpha ablation: frontier figure + table ----------
apts={}
# alpha=0.8 (base ICO seed) and alpha=1.0 (base PtO seed) from base seed CSV
bseed=sorted(glob.glob(BASE_GLOB))
bseed=[f for f in bseed if f"seed{SEED}." in f] or bseed[:1]
if bseed:
    r={x["model"]:x for x in csv.DictReader(open(bseed[0]))}
    if "ICO" in r: apts[0.80]={k:float(r["ICO"][kk]) for k,kk in [("mae","mae"),("curt","avg_curtailment"),("bias","bias")]}
    if "PtO" in r: apts[1.00]={k:float(r["PtO"][kk]) for k,kk in [("mae","mae"),("curt","avg_curtailment"),("bias","bias")]}
for d in sorted(glob.glob(os.path.join(ALPHA_ROOT,"a*"))):
    try: a=float(os.path.basename(d)[1:])
    except: continue
    fs=glob.glob(os.path.join(d,"**",f"comparison_seed{SEED}.csv"),recursive=True)
    if not fs: continue
    r={x["model"]:x for x in csv.DictReader(open(fs[0]))}
    if "ICO" in r: apts[a]={"mae":float(r["ICO"]["mae"]),"curt":float(r["ICO"]["avg_curtailment"]),"bias":float(r["ICO"]["bias"])}
if len(apts)>=3:
    al=sorted(apts); usable=[a for a in al if apts[a]["mae"]<1000]  # drop alpha=0 collapse from axis
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    xs=[apts[a]["mae"] for a in usable]; ys=[apts[a]["curt"] for a in usable]
    ax.plot(xs,ys,"-",color="0.6",lw=1.2,zorder=1)
    for a in usable:
        op=abs(a-0.8)<1e-9
        ax.scatter(apts[a]["mae"],apts[a]["curt"],s=110 if op else 60,color="#0F6E56" if op else "#BA7517",
                   edgecolor="white",lw=1,zorder=3)
        ax.annotate(f"$\\alpha$={a:.1f}"+("  (operating point)" if op else ""),(apts[a]["mae"],apts[a]["curt"]),
                    textcoords="offset points",xytext=(8,5),fontsize=9.5,fontweight="bold" if op else "normal",
                    color="#0B5240" if op else "0.3")
    if 0.0 in apts:
        ax.annotate(f"$\\alpha$=0: forecast collapses\n(MAE\u2248{apts[0.0]['mae']:.0f}, curtailment\u2248{apts[0.0]['curt']:.0f})",
                    (0.97,0.95),xycoords="axes fraction",ha="right",va="top",fontsize=8.5,color="#8a2a14",
                    bbox=dict(boxstyle="round,pad=0.4",fc="#fdf0ec",ec="none"))
    ax.set_xlabel("Forecast error \u2014 MAE (MW)  [accuracy: lower better]")
    ax.set_ylabel("Avg curtailment (MW)  [decision quality: lower better]")
    ax.set_title("$\\alpha$-sensitivity frontier: accuracy vs curtailment",fontsize=11,loc="left")
    ax.grid(True,color="0.92",lw=.6)
    for s in("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/alpha_frontier.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{OUT}/alpha_frontier.pdf",bbox_inches="tight")
    plt.close(fig); print("saved alpha_frontier")
    # alpha table
    pto_c=apts.get(1.00,{}).get("curt",np.nan)
    rowsA=[]
    for a in al:
        red=(pto_c-apts[a]["curt"])/pto_c*100 if pto_c else np.nan
        note="collapse" if apts[a]["mae"]>1000 else (f"\u2212{red:.0f}%" if a!=1.0 else "\u2014")
        rowsA.append([f"{a:.1f}"+("  (PtO)" if a==1.0 else ""),f"{apts[a]['mae']:.1f}",f"{apts[a]['curt']:.1f}",
                      f"{apts[a]['bias']:+.0f}",note])
    _table(rowsA,["$\\alpha$","MAE (MW)","Curtailment (MW)","Bias (MW)","Curt. vs PtO"],
           "Table 12.  Loss-weight $\\alpha$ ablation (seed "+SEED+")","table12_alpha",[.16,.20,.24,.18,.18])
else:
    print(f"!! alpha frontier needs >=3 points (found {sorted(apts)}); run the alpha sweep")

print(f"\nALL OUTPUTS -> {OUT}/")
