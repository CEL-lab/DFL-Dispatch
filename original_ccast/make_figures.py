#!/usr/bin/env python
# =====================================================================
# make_figures.py — paper figures from the 5-seed ICO results.
#
# Run from the work dir (where data/ and results/ico_model/ live):
#     pip install matplotlib        # once, in the e2e venv
#     python make_figures.py
#
# Reads results/ico_model/{metrics,history,dispatch_matrix,pred_ico,y_test}*
# and data/modified_dataset.csv. Writes PNG+PDF to results/ico_model/figures/.
# =====================================================================
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS  = os.environ.get("SAVE_DIR", "results/ico_model")
DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT      = os.path.join(RESULTS, "figures"); os.makedirs(OUT, exist_ok=True)
BEST_BY  = os.environ.get("BEST_BY", "test_loss")   # test_loss | r2 | avg_curtailment

TARGET   = "total load actual"
GEN_COLS = [
    "generation biomass", "generation fossil brown coal/lignite",
    "generation fossil gas", "generation fossil hard coal", "generation fossil oil",
    "generation hydro run-of-river and poundage", "generation hydro water reservoir",
    "generation nuclear", "generation other", "generation other renewable",
    "generation solar", "generation waste", "generation wind onshore",
]
GEN_LABELS  = ["Biomass","Coal","Gas","Hard Coal","Oil","Hydro ROR","Hydro Res",
               "Nuclear","Other","Oth.Renew","Solar","Waste","Wind"]
FOSSIL_LABELS = ["Coal","Gas","Hard Coal","Oil"]

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.7, "figure.dpi": 300,
})
C_TR, C_VA, C_FOSSIL, C_CLEAN, C_CAP = "#1b4f72", "#c0392b", "#b03a2e", "#1e8449", "#aeb6bf"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig); print("  saved", name)

# ---------------------------------------------------------------- metrics
mfiles = sorted(glob.glob(os.path.join(RESULTS, "metrics_seed*.csv")))
assert mfiles, f"no metrics_seed*.csv in {RESULTS}"
M = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True).sort_values("seed").reset_index(drop=True)
num = M.select_dtypes("number")
summary = pd.concat([num.mean().rename("mean"), num.std().rename("std")], axis=1).T
summary.to_csv(os.path.join(RESULTS, "metrics_summary.csv"))
best_seed = int(M.loc[M["r2"].idxmax(), "seed"]) if BEST_BY == "r2" \
            else int(M.loc[M[BEST_BY].idxmin(), "seed"])
print("Per-seed:\n", M[["seed","r2","mae","bias","avg_curtailment","test_loss"]].to_string(index=False))
print(f"\nBest seed by {BEST_BY}: {best_seed}")
print(f"Mean R2={num['r2'].mean():.4f}  MAE={num['mae'].mean():.1f}  "
      f"Bias={num['bias'].mean():+.1f}  Curtail={num['avg_curtailment'].mean():.1f}")

H = {int(s): np.load(os.path.join(RESULTS, f"history_seed{s}.npy"), allow_pickle=True).item()
     for s in M["seed"]}
hb = H[best_seed]
ep = np.array(hb["epoch"])

# ---------------------------------------------------------------- Fig 1: loss
def fig_loss():
    tr, va = np.array(hb["train_loss"]), np.array(hb["val_loss"])
    wu = int(np.argmax(np.array(hb["oc_train"]) > 0))  # first QP-active epoch index
    best = int(ep[np.argmin(va)])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axvspan(0.5, ep[wu] - 0.5, color="#7f8c8d", alpha=0.10, lw=0)
    ax.text((ep[wu]) / 2, ax.get_ylim()[1], "warm-up\n(MSE only)", ha="center", va="top",
            fontsize=9, color="#555")
    ax.plot(ep, tr, color=C_TR, lw=1.9, label="Training loss")
    ax.plot(ep, va, color=C_VA, lw=1.9, label="Validation loss")
    ax.axvline(best, color="#7f8c8d", ls=(0, (4, 3)), lw=1.1)
    ax.scatter([best], [va.min()], color=C_VA, s=40, zorder=5, ec="white", lw=0.8)
    ax.annotate(f"best epoch ({best})", xy=(best, va.min()), xytext=(best - 3, va.min() * 1.9),
                fontsize=9, ha="right", arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.set_yscale("log"); ax.set_xlim(1, ep.max())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Composite loss  $\alpha\,\mathrm{MSE}+(1-\alpha)\,\mathrm{OC}/\kappa$")
    ax.set_title(f"ICO: training vs. validation loss (seed {best_seed})", fontsize=12.5, pad=10)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig1_train_val_loss")

# ---------------------------------------------------------------- Fig 2: OC
def fig_oc():
    oct_, ocv = np.array(hb["oc_train"]), np.array(hb["oc_val"])
    m = oct_ > 0                       # QP-active epochs only
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ep[m], oct_[m], color=C_TR, lw=1.9, label="Training OC")
    if np.any(ocv[m] > 0):
        ax.plot(ep[m], ocv[m], color=C_VA, lw=1.9, label="Validation OC")
    ax.set_xlim(ep[m].min(), ep.max())
    ax.set_xlabel("Epoch"); ax.set_ylabel("Operational cost  (model units)")
    ax.set_title(f"ICO: operational cost during training (seed {best_seed})", fontsize=12.5, pad=10)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig2_operational_cost")

# ---------------------------------------------------------------- installed capacity
def installed_capacity():
    de = pd.read_csv(os.path.join(DATA_DIR, "data", "modified_dataset.csv"))
    de["time"] = pd.to_datetime(de["time"], utc=True)
    dw = pd.read_csv(os.path.join(DATA_DIR, "data", "weather_features.csv"))
    dw["time"] = pd.to_datetime(dw["dt_iso"], utc=True)
    dw = dw[["time","temp","humidity","wind_speed","clouds_all"]].groupby("time").mean().reset_index()
    df = pd.merge(de, dw, on="time", how="inner").dropna(subset=[TARGET] + GEN_COLS + ["temp"])
    tr_end = int(len(df) * 0.8)
    return df[GEN_COLS].iloc[:tr_end].max().values.astype(float)   # MW

# ---------------------------------------------------------------- Fig 3: cap vs dispatch
def fig_cap_dispatch():
    D = np.load(os.path.join(RESULTS, f"dispatch_matrix_seed{best_seed}.npy"))  # [N,13] MW
    disp = D.mean(axis=0)
    cap = installed_capacity()
    order = np.argsort(cap)[::-1]
    y = np.arange(len(order))
    colors = [C_FOSSIL if GEN_LABELS[i] in FOSSIL_LABELS else C_CLEAN for i in order]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.barh(y, cap[order], color=C_CAP, label="Installed capacity", zorder=1)
    ax.barh(y, disp[order], color=colors, label="Mean ICO dispatch", zorder=2, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([GEN_LABELS[i] for i in order])
    ax.invert_yaxis(); ax.set_xlabel("Power  (MW)")
    ax.set_title(f"Installed capacity vs. mean dispatch by source (seed {best_seed})",
                 fontsize=12.5, pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_CAP, label="Installed capacity"),
                       Patch(color=C_FOSSIL, label="Dispatch — fossil"),
                       Patch(color=C_CLEAN, label="Dispatch — clean")],
              frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0)
    save(fig, "fig3_capacity_vs_dispatch")

# ---------------------------------------------------------------- Fig 4: all-seed band
def fig_loss_band():
    L = min(len(H[s]["epoch"]) for s in H)
    e = np.arange(1, L + 1)
    TR = np.vstack([np.array(H[s]["train_loss"])[:L] for s in H])
    VA = np.vstack([np.array(H[s]["val_loss"])[:L] for s in H])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arr, c, lab in [(TR, C_TR, "Training"), (VA, C_VA, "Validation")]:
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(e, mu, color=c, lw=1.9, label=f"{lab} (mean)")
        ax.fill_between(e, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
    ax.set_yscale("log"); ax.set_xlim(1, L)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Composite loss  $\alpha\,\mathrm{MSE}+(1-\alpha)\,\mathrm{OC}/\kappa$")
    ax.set_title("ICO learning curve — mean ± 1 std across 5 seeds", fontsize=12.5, pad=10)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig4_learning_curve_allseeds")

# ---------------------------------------------------------------- Fig 5: pred vs actual
def fig_pred_actual():
    pred = np.load(os.path.join(RESULTS, f"pred_ico_seed{best_seed}.npy"))
    yt = np.load(os.path.join(RESULTS, "y_test.npy"))
    n = min(len(pred), len(yt)); pred, yt = pred[:n], yt[:n]
    s = max(0, n // 2 - 84); wk = slice(s, s + 168)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1.plot(yt[wk], color="#34495e", lw=1.6, label="Actual")
    a1.plot(pred[wk], color=C_VA, lw=1.6, ls="--", label="ICO forecast")
    a1.set_xlabel("Hour (sample week)"); a1.set_ylabel("Load (MW)")
    a1.set_title("Forecast vs. actual"); a1.legend(frameon=False)
    a2.scatter(yt, pred, s=6, alpha=0.25, color=C_TR, ec="none")
    lo, hi = min(yt.min(), pred.min()), max(yt.max(), pred.max())
    a2.plot([lo, hi], [lo, hi], color="#7f8c8d", lw=1.2, ls=(0, (4, 3)))
    r2 = 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)
    a2.set_xlabel("Actual load (MW)"); a2.set_ylabel("Predicted load (MW)")
    a2.set_title(f"Calibration  (R$^2$={r2:.4f})"); a2.set_aspect("equal", "box")
    fig.suptitle(f"ICO predictions on the test set (seed {best_seed})", fontsize=12.5)
    save(fig, "fig5_pred_vs_actual")

# ---------------------------------------------------------------- Fig 6: dispatch stack
def fig_dispatch_stack():
    D = np.load(os.path.join(RESULTS, f"dispatch_matrix_seed{best_seed}.npy"))
    yt = np.load(os.path.join(RESULTS, "y_test.npy"))[:len(D)]
    s = max(0, len(D) // 2 - 84); wk = slice(s, s + 168)
    order = np.argsort(D.mean(0))[::-1]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.stackplot(np.arange(168), *[D[wk, i] for i in order],
                 labels=[GEN_LABELS[i] for i in order],
                 colors=[cmap(k % 20) for k in range(len(order))])
    ax.plot(np.arange(168), yt[wk], color="black", lw=1.4, label="Actual load")
    ax.set_xlim(0, 167); ax.set_xlabel("Hour (sample week)"); ax.set_ylabel("Power (MW)")
    ax.set_title(f"ICO dispatch composition over a sample week (seed {best_seed})",
                 fontsize=12.5, pad=10)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0)
    save(fig, "fig6_dispatch_stack_week")

print("\nGenerating figures ->", OUT)
for fn in [fig_loss, fig_oc, fig_cap_dispatch, fig_loss_band, fig_pred_actual, fig_dispatch_stack]:
    try:
        fn()
    except Exception as e:
        print(f"  [skip {fn.__name__}] {type(e).__name__}: {e}")
print("\nDone. Aggregated table -> metrics_summary.csv")
