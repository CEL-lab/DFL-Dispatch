#!/usr/bin/env python3
"""
alpha-sensitivity frontier (supports Contribution #2).

Reads the per-alpha ICO results produced by run_alpha.pbs and plots the
accuracy-vs-curtailment trade-off, marking alpha=0.80 as the operating point.
Pulls the ICO row from each results/alpha/a<ALPHA>/seed<SEED>/comparison_seed*.csv,
averages over seeds, and (optionally) folds in your existing base/PtO points.

Run (post-hoc, no GPU):
    python alpha_frontier.py            # auto-discovers results/alpha/
"""
import os, csv, glob, re, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_ROOT = os.environ.get("ALPHA_ROOT", "results/alpha")

# Known points you already have (optional convenience so you needn't rerun them):
#   alpha=0.80 -> base ICO ;  alpha=1.00 -> PtO (MSE only)
KNOWN = {
    0.80: {"mae": 353.8, "r2": 0.9874, "curt": 134.6, "oc": 179.9},
    1.00: {"mae": 350.9, "r2": 0.9877, "curt": 189.7, "oc": 234.2},
}

def load_ico_rows(root):
    pts = {}
    for d in sorted(glob.glob(os.path.join(root, "a*"))):
        m = re.search(r"a([0-9.]+)", os.path.basename(d))
        if not m: continue
        a = float(m.group(1))
        rows = []
        for f in glob.glob(os.path.join(d, "seed*", "comparison_seed*.csv")):
            for r in csv.DictReader(open(f)):
                if r["model"] == "ICO":
                    rows.append({k: float(v) for k, v in r.items() if k not in ("model", "seed")})
        if rows:
            pts[a] = {"mae": st.mean(x["mae"] for x in rows),
                      "r2":  st.mean(x["r2"] for x in rows),
                      "curt":st.mean(x["avg_curtailment"] for x in rows),
                      "oc":  st.mean(x["avg_operational_cost"] for x in rows)}
    return pts

pts = dict(KNOWN)
pts.update(load_ico_rows(ALPHA_ROOT))   # measured runs override the known stubs
alphas = sorted(pts)

print(f"{'alpha':>6} | {'MAE':>7} {'R2':>8} {'curtail':>9} {'OC':>8}")
for a in alphas:
    p = pts[a]; print(f"{a:6.2f} | {p['mae']:7.1f} {p['r2']:8.4f} {p['curt']:9.1f} {p['oc']:8.1f}")

# ---- frontier figure: accuracy (MAE) vs decision quality (curtailment) ----
mae  = [pts[a]["mae"] for a in alphas]
curt = [pts[a]["curt"] for a in alphas]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(mae, curt, "-", color="0.6", lw=1.2, zorder=1)
for a, x, y in zip(alphas, mae, curt):
    knee = abs(a - 0.80) < 1e-9
    ax.scatter(x, y, s=90 if knee else 55,
               color="#0F6E56" if knee else "#BA7517",
               edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(f"\u03b1={a:.2f}" + ("  (operating point)" if knee else ""),
                (x, y), textcoords="offset points", xytext=(8, 5),
                fontsize=9, fontweight="bold" if knee else "normal",
                color="#0B5240" if knee else "0.3")
ax.set_xlabel("Forecast error  \u2014  MAE (MW)   [accuracy: lower better]")
ax.set_ylabel("Avg. curtailment (MW)   [decision quality: lower better]")
ax.set_title("\u03b1-sensitivity frontier: accuracy vs curtailment", fontsize=11, loc="left")
ax.grid(True, color="0.92", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.annotate("more MSE-focused\n(\u03b1\u21921, = PtO)", (0.02, 0.95), xycoords="axes fraction",
            fontsize=8, color="0.45", va="top")
ax.annotate("more decision-focused\n(\u03b1\u21920)", (0.98, 0.06), xycoords="axes fraction",
            fontsize=8, color="0.45", ha="right")
fig.tight_layout()
fig.savefig("alpha_frontier.png", dpi=200, bbox_inches="tight")
fig.savefig("alpha_frontier.pdf", bbox_inches="tight")
print("\nSaved alpha_frontier.png / .pdf")
