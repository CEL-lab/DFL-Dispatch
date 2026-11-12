#!/usr/bin/env python
"""Paper figures from run_experiment.py outputs (vector PDF + PNG).

Usage:  python figures.py <results_root> <out_dir> [--data modified_dataset.csv]
  results_root: folder with <tag>/seed<S>/{results.json,hourly_outputs.npz,history_*.npy}
Figures (each skipped if its inputs are absent):
  fig_training        ICO composite loss and operational cost per epoch, mean and seed band
  fig_comparison      MAE and modeled shortfall for PtO / ICO / ASYM / OFFSET / LS, seed error bars
  fig_alpha_frontier  validation and test MAE vs shortfall over alpha (ICO), with PtO and OFFSET
  fig_conditions      shortfall by condition (base, S2, S3, rolling years) for PtO / ICO / OFFSET
  fig_dispatch        ICO mean dispatch per source vs realized generation on the test hours
Colours: validated categorical palette (dataviz skill, light mode), one fixed hue per model.
"""
import sys, os, glob, json, argparse, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--data", default=None, help="modified_dataset.csv, for realized generation if the npz lacks it")
ap.add_argument("--base", default="base_a0.8_avphysical", help="tag for the training and comparison figures")
ap.add_argument("--dispatch-base", default="base_a0.8", help="tag for the dispatch-vs-realized figure (default: proxy ceiling)")
ap.add_argument("--suffix", default="_avphysical", help="availability suffix of the sweep tags used for the alpha and conditions figures ('' for proxy)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

COL = {"PtO": "#2a78d6", "ICO": "#eb6834", "ASYM": "#1baf7a", "OFFSET": "#eda100", "LS": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
ORDER = ["PtO", "ICO", "ASYM", "OFFSET", "LS"]
LABEL = {"PtO": "PtO", "ICO": "ICO", "ASYM": "ASYM (closed form)", "OFFSET": "OFFSET", "LS": "LS"}

def load(tag):
    out = collections.defaultdict(lambda: collections.defaultdict(list)); files = sorted(glob.glob(os.path.join(a.root, tag, "seed*", "results.json")))
    for f in files:
        r = json.load(open(f))
        for m, v in r["models"].items():
            for k, x in v["test"].items():
                if isinstance(x, (int, float)): out[m]["test_" + k].append(x)
            for k, x in v["val"].items(): out[m]["val_" + k].append(x)
    return out if files else None

def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", name)

def ms(d, k): x = np.asarray(d[k], float); return x.mean(), (x.std() if len(x) > 1 else 0.0)

# ---------------- fig_training ----------------
hist = [np.load(f, allow_pickle=True).item() for f in sorted(glob.glob(os.path.join(a.root, a.base, "seed*", "history_ICO.npy")))]
if hist:
    L = min(len(h["epoch"]) for h in hist)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for i, (key_tr, key_va, ttl, ylab) in enumerate([("train_mse", "val_loss", "Composite loss", "loss (standardised units)"),
                                                     ("train_oc", "val_oc", "Operational cost", "OC (dimensionless, GW units)")]):
        ep = np.array(hist[0]["epoch"][:L])
        for key, lab, c in [(key_tr, "training", COL["PtO"]), (key_va, "validation", COL["ICO"])]:
            M = np.array([h[key][:L] for h in hist], float); mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
            ax[i].plot(ep, mu, lw=2, color=c, label=lab); ax[i].fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
        ax[i].axvspan(0.5, 5.5, color=GRID, alpha=0.6, lw=0)
        ax[i].text(0.08, 0.03, "MSE\nwarm-up", transform=ax[i].transAxes, ha="left", va="bottom", fontsize=6.5, color=INK2)
        ax[i].set_title(ttl, loc="left", fontsize=9, color=INK); ax[i].set_xlabel("epoch"); ax[i].set_ylabel(ylab)
        if i == 0: ax[i].set_yscale("log")
    ax[0].legend(loc="upper right", fontsize=8); ax[1].legend(loc="upper right", fontsize=8); fig.subplots_adjust(wspace=0.35); save(fig, "fig_training")

# ---------------- fig_comparison ----------------
base = load(a.base)
if base:
    models = [m for m in ORDER if m in base]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, (k, ttl) in enumerate([("test_mae", "Forecast error, MAE (MW)"), ("test_shortfall_mw", "Mean modeled shortfall (MW)")]):
        vals = [ms(base[m], k) for m in models]; x = np.arange(len(models))
        ax[i].bar(x, [v[0] for v in vals], width=0.62, color=[COL[m] for m in models], yerr=[v[1] for v in vals],
                  error_kw=dict(ecolor=INK2, lw=1, capsize=3), edgecolor="none")
        for xi, v in zip(x, vals): ax[i].text(xi, v[0] + v[1] + ax[i].get_ylim()[1] * 0.01, f"{v[0]:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax[i].set_xticks(x); ax[i].set_xticklabels(models); ax[i].set_title(ttl, loc="left", fontsize=9); ax[i].grid(axis="x", visible=False)
    save(fig, "fig_comparison")

# ---------------- fig_alpha_frontier ----------------
tags = sorted(glob.glob(os.path.join(a.root, "base_a*")))
alphas = []
for t in tags:
    nm = os.path.basename(t)
    if "testyear" in nm or "_hr" in nm or "_fl" in nm: continue
    if a.suffix and not nm.endswith(a.suffix): continue
    if not a.suffix and "_av" in nm: continue
    alphas.append((float(nm.split("_a")[1].split("_")[0]), load(nm)))
if len(alphas) >= 3:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    offscale = [(al, d) for al, d in alphas if al < 0.3]; alphas = [(al, d) for al, d in alphas if al >= 0.3]   # alpha=0 is off-scale
    for i, (pre, ttl) in enumerate([("val_", "Validation"), ("test_", "Test")]):
        for al, d in offscale:
            if "ICO" in d:
                ax[i].text(0.98, 0.02, f"α={al:g} (off scale): MAE {ms(d['ICO'], pre + 'mae')[0]:,.0f} MW, shortfall {ms(d['ICO'], pre + 'shortfall_mw')[0]:.0f} MW",
                           transform=ax[i].transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
        for al, d in sorted(alphas):
            if "ICO" not in d: continue
            x, sx = ms(d["ICO"], pre + "mae"); y, sy = ms(d["ICO"], pre + "shortfall_mw")
            ax[i].errorbar(x, y, xerr=sx, yerr=sy, fmt="o", ms=5, color=COL["ICO"], ecolor=INK2, elinewidth=0.8, capsize=2)
            ax[i].annotate(f"α={al:g}", (x, y), xytext=(-7, 3), textcoords="offset points", ha="right", fontsize=7.5, color=INK)
        pts = sorted([(ms(d["ICO"], pre + "mae")[0], ms(d["ICO"], pre + "shortfall_mw")[0]) for al, d in alphas if "ICO" in d])
        ax[i].plot([p[0] for p in pts], [p[1] for p in pts], color=COL["ICO"], lw=1.2, alpha=0.6, zorder=0, label="ICO over α")
        if base:
            for m, mk, off in [("PtO", "s", (8, 2)), ("OFFSET", "D", (8, -3))]:
                if m in base:
                    x, sx = ms(base[m], pre + "mae"); y, sy = ms(base[m], pre + "shortfall_mw")
                    ax[i].errorbar(x, y, xerr=sx, yerr=sy, fmt=mk, ms=6, color=COL[m], ecolor=INK2, elinewidth=0.8, capsize=2, label=m)
                    ax[i].annotate(m, (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color=INK)
        ax[i].set_xlabel("MAE (MW)"); ax[i].set_ylabel("mean modeled shortfall (MW)"); ax[i].set_title(ttl, loc="left", fontsize=9)
    ax[0].legend(fontsize=7.5, loc="upper right"); fig.subplots_adjust(wspace=0.3); save(fig, "fig_alpha_frontier")

# ---------------- fig_conditions ----------------
conds = [("Base", a.base), ("S2 volatility", "s2_a0.8" + a.suffix), ("S3 tight capacity", "s3_a0.8" + a.suffix),
         ("Test 2016", "base_testyear2016_a0.8" + a.suffix), ("Test 2017", "base_testyear2017_a0.8" + a.suffix), ("Test 2018", "base_testyear2018_a0.8" + a.suffix)]
have = [(lab, load(t)) for lab, t in conds]; have = [(lab, d) for lab, d in have if d and "ICO" in d and "PtO" in d]
if len(have) >= 2:
    fig, ax = plt.subplots(figsize=(7.2, 2.8)); series = [m for m in ["PtO", "ICO", "OFFSET"] if all(m in d for _, d in have)]
    x = np.arange(len(have)); w = 0.8 / len(series)
    for j, m in enumerate(series):
        vals = [ms(d[m], "test_shortfall_mw") for _, d in have]
        ax.bar(x + (j - (len(series) - 1) / 2) * w, [v[0] for v in vals], width=w * 0.92, color=COL[m], label=LABEL[m],
               yerr=[v[1] for v in vals], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ji = series.index("ICO")
    for xi, (_, d) in zip(x, have):
        p, (i_, si) = ms(d["PtO"], "test_shortfall_mw")[0], ms(d["ICO"], "test_shortfall_mw")
        ax.text(xi + (ji - (len(series) - 1) / 2) * w, i_ + si + ax.get_ylim()[1] * 0.015, f"{100 * (i_ - p) / p:+.0f}%",
                ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([lab.replace(" ", "\n", 1) if " " in lab else lab for lab, _ in have])
    ax.set_ylabel("mean modeled shortfall (MW)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=1, loc="upper right", title="(% = ICO vs PtO)", title_fontsize=7.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08); save(fig, "fig_conditions")

# ---------------- fig_dispatch ----------------
npz = sorted(glob.glob(os.path.join(a.root, a.dispatch_base, "seed*", "hourly_outputs.npz")))
if npz:
    Z = [np.load(f, allow_pickle=True) for f in npz]
    r0 = json.load(open(os.path.join(os.path.dirname(npz[0]), "results.json")))
    labels = list(r0["models"]["ICO"]["test"]["per_source_mw"].keys())
    disp = np.mean([z["ICO_test_dispatch"].mean(0) for z in Z], 0); cap = np.mean([z["cap_test"].mean(0) for z in Z], 0)
    actual = None
    if "actual_gen_test" in Z[0].files: actual = Z[0]["actual_gen_test"].mean(0)
    elif a.data:
        import pandas as pd
        e = pd.read_csv(a.data); e["time"] = pd.to_datetime(e["time"], utc=True).astype(str)
        cols = ["generation biomass", "generation fossil brown coal/lignite", "generation fossil gas", "generation fossil hard coal", "generation fossil oil",
                "generation hydro run-of-river and poundage", "generation hydro water reservoir", "generation nuclear", "generation other",
                "generation other renewable", "generation solar", "generation waste", "generation wind onshore"]
        sel = e.set_index("time").reindex([str(t) for t in Z[0]["time_test"]])[cols]
        actual = sel.values.astype(float).mean(0) if sel.notna().all().all() else None
    order = np.argsort(disp)[::-1]; y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); h = 0.38
    ax.barh(y + h / 2, disp[order], height=h * 0.92, color=COL["ICO"], label="ICO mean dispatch")
    if actual is not None: ax.barh(y - h / 2, actual[order], height=h * 0.92, color=INK2, label="realized generation (ENTSO-E)")
    xmax = 1.25 * max(disp.max(), actual.max() if actual is not None else 0)
    capc = np.minimum(cap, xmax * 0.985)
    ax.scatter(capc[order], y + h / 2, marker="|", s=70, color=INK, label="mean capacity ceiling", zorder=1, alpha=0.7)
    for yi, k in zip(y, order):
        ax.text(disp[k] + xmax * 0.01, yi + h / 2, f"{disp[k]:.0f}", va="center", fontsize=7, color=INK, zorder=4,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
        if actual is not None: ax.text(actual[k] + xmax * 0.01, yi - h / 2, f"{actual[k]:.0f}", va="center", fontsize=7, color=INK2, zorder=4,
                                       bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
        if cap[k] > xmax: ax.text(xmax * 0.985, yi + h / 2, f"ceiling {cap[k]:,.0f} →", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlim(0, xmax); ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in order]); ax.invert_yaxis()
    ax.set_xlabel("MW, mean over test hours"); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right"); save(fig, "fig_dispatch")
print("done ->", a.out)
