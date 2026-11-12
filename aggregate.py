#!/usr/bin/env python
"""Aggregate run_experiment.py outputs across seeds.

Usage:  python aggregate.py <results_root> [out_prefix]
  results_root: folder containing <tag>/seed<S>/results.json (e.g. ccast_results_v2/results/v2)
Writes <out_prefix>_summary.csv (one row per tag x model, mean/std over seeds) and
<out_prefix>_summary.md (readable tables), and prints the tables.
If hourly_outputs.npz is present, also computes paired, dependence-aware uncertainty for
ICO-PtO, ICO-ASYM and ICO-OFFSET mean-shortfall differences with a moving-block bootstrap
over test hours (block = 168 h, 2000 resamples, seeds pooled).
"""
import sys, os, glob, json, collections
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "..", os.path.basename(os.path.normpath(root)))
TEST_KEYS = ["r2", "mae", "rmse", "mape", "bias", "shortfall_mw", "shortfall_rate", "shortfall_mwh", "excess_mw",
             "op_cost", "qp_obj", "slack_hours", "identity_residual_mw", "fossil_mw", "clean_mw"]
VAL_KEYS = ["mae", "bias", "shortfall_mw", "shortfall_rate", "op_cost", "mse_std"]
TRAIN_KEYS = ["train_minutes", "kappa", "best_epoch", "epochs_run", "qp_failed_batches", "offset_mw"]
ORDER = ["PtO", "ICO", "ASYM", "OFFSET", "LS"]

rows = collections.defaultdict(lambda: collections.defaultdict(list))
meta = {}
for f in sorted(glob.glob(os.path.join(root, "*", "seed*", "results.json"))):
    r = json.load(open(f)); tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
    meta.setdefault(tag, dict(n_test=r["split"]["n_test"], n_usable=r["split"]["n_usable"], test_start=r["split"]["test_start"],
                              check=r["checks"]["qp_vs_closed"], s3=r["config"].get("s3"), env=r["env"]))
    for m, v in r["models"].items():
        d = rows[(tag, m)]
        d["seed"].append(r["config"]["seed"])
        for k in TEST_KEYS: d["test_" + k].append(v["test"].get(k, np.nan))
        for k in VAL_KEYS: d["val_" + k].append(v["val"].get(k, np.nan))
        for k in TRAIN_KEYS:
            x = v.get("train", {}).get(k)
            if x is not None: d["train_" + k].append(x)
        st = v["test"].get("qp_status", {})
        d["test_inaccurate"].append(st.get("optimal_inaccurate", 0)); d["test_fallbacks"].append(st.get("closed_form_fallbacks", 0))

def fmt(a, d=1):
    a = np.asarray(a, float)
    if len(a) == 0 or np.all(np.isnan(a)): return "-"
    return f"{np.nanmean(a):.{d}f} ± {np.nanstd(a):.{d}f}" if len(a) > 1 else f"{a[0]:.{d}f}"

# ---- block bootstrap on hourly shortfall differences ----
def block_bootstrap(diffs, block=168, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(diffs); nb = int(np.ceil(n / block)); out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = diffs[idx].mean()
    return np.percentile(out, [2.5, 97.5])

boot = {}
for tag in meta:
    files = sorted(glob.glob(os.path.join(root, tag, "seed*", "hourly_outputs.npz")))
    if not files: continue
    short = collections.defaultdict(list)
    for f in files:
        z = np.load(f)
        for m in ORDER:
            k = f"{m}_test_shortfall"
            if k in z: short[m].append(z[k])
    if "ICO" not in short: continue
    for other in ["PtO", "ASYM", "OFFSET"]:
        if other not in short: continue
        d = np.concatenate([a - b for a, b in zip(short["ICO"], short[other])])  # pooled over seeds, hour-paired
        lo, hi = block_bootstrap(d)
        boot[(tag, other)] = (d.mean(), lo, hi)

# ---- write ----
lines = []; csv = ["tag,model,n_seeds," + ",".join(f"{k}_mean,{k}_std" for k in
                   ["test_" + k for k in TEST_KEYS] + ["val_" + k for k in VAL_KEYS] + ["train_" + k for k in TRAIN_KEYS])]
for tag in sorted(meta):
    mt = meta[tag]
    lines.append(f"\n## {tag}  (n_test = {mt['n_test']}, n_usable = {mt['n_usable']}, test starts {mt['test_start']})")
    lines.append(f"QP-vs-closed-form check: {mt['check']}" + (f"; S3: {mt['s3']}" if mt["s3"] else ""))
    lines.append("")
    lines.append("| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in ORDER:
        d = rows.get((tag, m))
        if not d: continue
        extra = f" (offset {fmt(d['train_offset_mw'])} MW)" if "train_offset_mw" in d else ""
        lines.append(f"| {m}{extra} | {len(d['seed'])} | {fmt(d['test_mae'])} | {fmt(d['test_bias'])} | {fmt(d['test_shortfall_mw'])} | "
                     f"{fmt(d['test_shortfall_rate'])} | {fmt(d['test_excess_mw'])} | {fmt(d['test_op_cost'])} | {fmt(d['test_slack_hours'],0)} | "
                     f"{max(abs(x) for x in d['test_identity_residual_mw']):.3f} | {fmt(d['val_mae'])} | {fmt(d['val_shortfall_mw'])} | "
                     f"{fmt(d.get('train_train_minutes', []))} | {fmt(d.get('train_kappa', []),0)} | {fmt(d.get('train_best_epoch', []),0)} | {sum(d.get('train_qp_failed_batches', [0]))} |")
        vals = []
        for k in ["test_" + k for k in TEST_KEYS] + ["val_" + k for k in VAL_KEYS] + ["train_" + k for k in TRAIN_KEYS]:
            a = np.asarray(d.get(k, []), float); vals += [f"{np.nanmean(a):.6g}" if len(a) else "", f"{np.nanstd(a):.6g}" if len(a) > 1 else ""]
        csv.append(f"{tag},{m},{len(d['seed'])}," + ",".join(vals))
    bb = [(o, boot[(tag, o)]) for o in ["PtO", "ASYM", "OFFSET"] if (tag, o) in boot]
    if bb:
        lines.append("")
        lines.append("Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): "
                     + "; ".join(f"vs {o}: {m:.1f} [{lo:.1f}, {hi:.1f}]" for o, (m, lo, hi) in bb))
with open(out + "_bootstrap.csv", "w") as fh:
    fh.write("tag,other,mean_diff_mw,ci_lo,ci_hi\n")
    for (tag, o), (m_, lo, hi) in sorted(boot.items()): fh.write(f"{tag},{o},{m_:.3f},{lo:.3f},{hi:.3f}\n")
with open(out + "_meta.json", "w") as fh:
    json.dump({t: dict(n_test=m["n_test"], n_usable=m["n_usable"], test_start=m["test_start"], s3=m["s3"]) for t, m in meta.items()}, fh, indent=1)
env = next(iter(meta.values()))["env"]
header = [f"# Aggregated results for `{root}`", "",
          f"Environment of first run: " + ", ".join(f"{k} {env.get(k)}" for k in ["python", "torch", "cvxpy", "cvxpylayers", "diffcp", "scs", "ecos", "cuda"] if env.get(k)), ""]
md = "\n".join(header + lines) + "\n"
open(out + "_summary.md", "w").write(md); open(out + "_summary.csv", "w").write("\n".join(csv) + "\n")
print(md); print(f"wrote {out}_summary.md and {out}_summary.csv")
