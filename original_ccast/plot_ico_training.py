#!/usr/bin/env python3
"""
Figure 2 + OC curve — ICO training dynamics.

Reads the per-epoch history that three_model_comparison.py already saves
(history_ICO_seed{SEED}.npy) and plots:
  (a) train/validation composite loss vs epoch, with the 5-epoch MSE warm-up
      shaded and the QP-activation point (epoch 6) marked;
  (b) the operational-cost component (OC) vs epoch, which is ~0 during warm-up
      and becomes active once the dispatch layer engages.

No re-training and no code changes required.

Usage:
  SAVE_DIR=results/comparison SEED=42 python plot_ico_training.py
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = os.environ.get("SAVE_DIR", "results/comparison")
SEED     = os.environ.get("SEED", "42")
WARMUP   = 5                      # MSE-only warm-up epochs; QP activates at epoch 6

path = os.path.join(SAVE_DIR, f"history_ICO_seed{SEED}.npy")
if not os.path.exists(path):
    sys.exit(f"ERROR: {path} not found. Run with REUSE_ICO=0 so ICO history is saved, "
             f"or point SAVE_DIR at an alpha run dir (results/alpha/a0.8/seed{SEED}).")
h = np.load(path, allow_pickle=True).item()
ep = np.array(h["epoch"])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) composite loss
axL.axvspan(0, WARMUP + 0.5, color="0.92", lw=0, label="MSE warm-up")
axL.plot(ep, h["train_loss"], color="#0F6E56", lw=1.8, label="train")
axL.plot(ep, h["val_loss"],   color="#BA7517", lw=1.8, label="validation")
axL.axvline(WARMUP + 0.5, color="0.4", ls="--", lw=1)
axL.text(WARMUP + 0.7, max(h["train_loss"]) * 0.6, "QP activated\n(epoch 6)",
         fontsize=8.5, color="0.3")
axL.set_xlabel("Epoch"); axL.set_ylabel("Composite loss  $\\mathcal{L}$")
axL.set_title("(a) Training / validation convergence", fontsize=10.5, loc="left")
axL.set_yscale("log")
axL.legend(frameon=False, fontsize=9)
axL.grid(True, color="0.92", lw=0.6)
for s in ("top", "right"): axL.spines[s].set_visible(False)

# (b) OC component
axR.axvspan(0, WARMUP + 0.5, color="0.92", lw=0)
axR.plot(ep, h["oc_train"], color="#0F6E56", lw=1.8, label="train OC")
axR.plot(ep, h["oc_val"],   color="#BA7517", lw=1.8, label="val OC")
axR.axvline(WARMUP + 0.5, color="0.4", ls="--", lw=1)
axR.set_xlabel("Epoch"); axR.set_ylabel("Operational cost  OC$(z,y)$")
axR.set_title("(b) Operational-cost component", fontsize=10.5, loc="left")
axR.legend(frameon=False, fontsize=9)
axR.grid(True, color="0.92", lw=0.6)
for s in ("top", "right"): axR.spines[s].set_visible(False)

fig.suptitle(f"ICO training dynamics (seed {SEED}): stable QP integration, no spike at activation",
             fontsize=10.5, y=1.02)
fig.tight_layout()
out = os.path.join(SAVE_DIR, f"fig_ico_training_seed{SEED}")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png / .pdf")
print(f"final composite loss: train {h['train_loss'][-1]:.4f}  val {h['val_loss'][-1]:.4f}")
