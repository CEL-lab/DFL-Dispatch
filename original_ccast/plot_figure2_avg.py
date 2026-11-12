#!/usr/bin/env python
# =====================================================================
# Figure 2, averaged across seeds (Dr. Le, pt. 5).
#
# Loads every per-seed training history written by ico_train_single_seed.py
# and plots the SEED-MEAN composite loss (a) and operational cost (b), with a
# shaded min-max band across seeds instead of a single representative seed.
#
# Default: averages the base run's seeds in RESULTS_DIR.
#   RESULTS_DIR=results/ico_model python plot_figure2_avg.py
#
# To average across rolling-origin FOLDS instead of seeds, point GLOB at the
# per-fold sub-folders (each holds one history file), e.g.:
#   GLOB="results/ico_model/fold*of3/history_seed42.npy" python plot_figure2_avg.py
# =====================================================================

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: safe on the cluster / no display
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/ico_model")
GLOB        = os.environ.get("GLOB", os.path.join(RESULTS_DIR, "history_seed*.npy"))
WARMUP_EP   = int(os.environ.get("WARMUP_EP", "5"))
OUT_PATH    = os.environ.get("OUT_PATH", os.path.join(RESULTS_DIR, "figure2_seed_avg.png"))

files = sorted(glob.glob(GLOB))
if not files:
    raise SystemExit(f"No history files matched: {GLOB}\n"
                     f"Run the training seeds first, or set GLOB / RESULTS_DIR.")
print(f"Averaging {len(files)} runs:")
for f in files:
    print("   ", f)

hists = [np.load(f, allow_pickle=True).item() for f in files]


def stack(key, drop_warmup=False):
    """Stack one history key across runs, truncating to the shortest run so
    epochs line up. Optionally drop the warm-up epochs (where OC is 0)."""
    start = WARMUP_EP if drop_warmup else 0
    L = min(len(h[key]) for h in hists)
    arr = np.array([h[key][start:L] for h in hists], dtype=float)   # [n_runs, L-start]
    ep = np.arange(start + 1, L + 1)
    return ep, arr


def band(ax, key, color, label, drop_warmup=False):
    ep, arr = stack(key, drop_warmup=drop_warmup)
    ax.plot(ep, arr.mean(0), color=color, lw=1.8, label=label)
    ax.fill_between(ep, arr.min(0), arr.max(0), color=color, alpha=0.15, linewidth=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# ── Panel (a): composite loss, train & validation ──
band(ax1, "train_loss", "tab:blue", "Training loss")
band(ax1, "val_loss",   "tab:red",  "Validation loss")
ax1.axvspan(1, WARMUP_EP, color="gray", alpha=0.12)
ax1.text(max(WARMUP_EP / 2, 1), ax1.get_ylim()[1], "warm-up\n(MSE only)",
         ha="center", va="top", fontsize=8, color="gray")
ax1.set_yscale("log")
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"Composite loss  $\alpha\,$MSE $+(1-\alpha)\,$OC$/\kappa$")
ax1.set_title("(a) Composite loss")
ax1.legend(fontsize=8, frameon=False)

# ── Panel (b): operational cost, train & validation (post-warmup) ──
band(ax2, "oc_train", "tab:blue", "Training OC",   drop_warmup=True)
band(ax2, "oc_val",   "tab:red",  "Validation OC", drop_warmup=True)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Operational cost (dimensionless)")
ax2.set_title("(b) Operational cost")
ax2.legend(fontsize=8, frameon=False)

fig.suptitle(f"ICO training diagnostics — mean over {len(files)} seeds "
             f"(shaded: min-max across seeds)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PATH, dpi=200)
print(f"\nSaved -> {OUT_PATH}")
