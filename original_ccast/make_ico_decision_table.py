#!/usr/bin/env python3
"""
Table 8 — ICO dispatch decision variables.

Builds the per-source decision-variable table from the saved dispatch matrix
(dispatch_ICO_seed{SEED}.npy) and the reconstructed capacity ceiling: average
dispatch z_j, available capacity, utilization, and the fossil/clean split, plus
the slack variable (curtailment) summary. Outputs a CSV and a rendered table
image.

Usage:
  DATA_DIR=. SAVE_DIR=results/comparison SEED=42 python make_ico_decision_table.py
"""
import os
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
FOSSIL = {1,2,3,4}
VARIABLE_RENEWABLE = {"generation solar","generation wind onshore",
    "generation hydro run-of-river and poundage","generation hydro water reservoir"}

D = np.load(os.path.join(SAVE_DIR, f"dispatch_ICO_seed{SEED}.npy"))
y_test = np.load(os.path.join(SAVE_DIR, "y_test.npy"))

# reconstruct capacity ceiling aligned to the test split
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
cap_test=cap[vl_end:][-len(D):]

disp=D.mean(0); avail=cap_test.mean(0)
util=np.where(avail>1, disp/avail*100, np.nan)

# ---- build dataframe ----
recs=[]
for j in np.argsort(-disp):
    recs.append({"Source":LABEL[j], "Type":"Fossil" if j in FOSSIL else "Clean",
                 "Avg dispatch (MW)":round(disp[j],1),
                 "Avail. cap (MW)":round(avail[j],1),
                 "Utilization (%)":("%.1f"%util[j]) if not np.isnan(util[j]) else "\u2014"})
tbl=pd.DataFrame(recs)
csv_path=os.path.join(SAVE_DIR,f"ico_decision_variables_seed{SEED}.csv")
tbl.to_csv(csv_path,index=False); print("saved",csv_path)

# slack / summary
slack=np.maximum(y_test - D.sum(1), 0.0)
clean_sh=disp[[j for j in range(13) if j not in FOSSIL]].sum()/disp.sum()*100
print(f"total dispatch {disp.sum():.0f} MW | clean {clean_sh:.1f}% | fossil {100-clean_sh:.1f}%")
print(f"slack (curtailment): mean {slack.mean():.1f} MW | incidence {(slack>1).mean()*100:.1f}% | "
      f"max {slack.max():.0f} MW | total {slack.sum():.0f} MWh")

# ---- render image ----
fig,ax=plt.subplots(figsize=(8.5,0.45*(len(tbl)+3))); ax.axis("off")
cells=tbl.values.tolist()
t=ax.table(cellText=cells,colLabels=tbl.columns,cellLoc="center",loc="center",
           colWidths=[0.26,0.14,0.22,0.20,0.18])
t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.5)
for (r,c),cell in t.get_celld().items():
    cell.set_edgecolor("none")
    if r==0: cell.set_text_props(weight="bold",color="white"); cell.set_facecolor("#0F6E56")
    else:
        is_fossil = cells[r-1][1]=="Fossil"
        cell.set_facecolor("#fbeee9" if is_fossil else ("#f4f7f6" if r%2 else "white"))
ax.set_title(f"Table 8.  ICO dispatch decision variables (seed {SEED}, 3,490 test hours)",
             fontsize=11,weight="bold",loc="left",pad=14)
fig.text(0.02,0.005,f"Total dispatch {disp.sum():.0f} MW  |  clean {clean_sh:.1f}% / fossil {100-clean_sh:.1f}%  |  "
         f"slack: mean {slack.mean():.1f} MW, incidence {(slack>1).mean()*100:.1f}%",
         fontsize=8,style="italic",color="0.35")
fig.tight_layout()
out=os.path.join(SAVE_DIR,f"tbl8_ico_decision_seed{SEED}")
fig.savefig(out+".png",dpi=200,bbox_inches="tight"); fig.savefig(out+".pdf",bbox_inches="tight")
print("saved",out+".png / .pdf")
