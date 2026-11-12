#!/usr/bin/env python
"""LaTeX tables for the manuscript from aggregate.py outputs.

Usage:  python make_tables.py <prefix> <out_dir> [--avail physical|proxy]
  <prefix> is the aggregate.py output prefix, e.g. ccast_results_v3/v3  (reads <prefix>_summary.csv,
  <prefix>_bootstrap.csv, <prefix>_meta.json). --avail selects which availability model the sweep
  tables (alpha, conditions, capgrid) are read from; the main table always uses the physical ceiling
  and the availability table shows all three.
Writes tab_main.tex, tab_alpha.tex, tab_avail.tex, tab_conditions.tex, tab_capgrid.tex, tab_split.tex,
numbers.tex (\\newcommand macros for in-text numbers).
"""
import sys, os, csv, json, argparse, collections
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("out"); ap.add_argument("--avail", default="physical")
ap.add_argument("--root", default=None, help="results root (default: <dir of prefix>/results/<basename of prefix>)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
ROOT = a.root or os.path.join(os.path.dirname(a.prefix), "results", os.path.basename(a.prefix))
import glob
def per_source(tag, model, label):
    vals = []
    for f in glob.glob(os.path.join(ROOT, tag, "seed*", "results.json")):
        r = json.load(open(f)); m = r["models"].get(model)
        if m: vals.append(m["test"]["per_source_mw"][label])
    return float(np.mean(vals)) if vals else np.nan
def g42(tag, key):
    f = os.path.join(ROOT, tag, "seed42", "results.json")
    if not os.path.exists(f): return np.nan
    r = json.load(open(f)); m = r["models"].get("ICO")
    return float(m["test"][key]) if m else np.nan
ALNAME = {1.0: "One", 0.95: "NinetyFive", 0.9: "Ninety", 0.8: "Eighty", 0.7: "Seventy", 0.5: "Fifty", 0.3: "Thirty", 0.0: "Zero"}
def alname(al): return ALNAME.get(al, "A" + str(al).replace(".", "p"))
SUF = "" if a.avail == "proxy" else f"_av{a.avail}"
rows = {(r["tag"], r["model"]): r for r in csv.DictReader(open(a.prefix + "_summary.csv"))}
boot = {(r["tag"], r["other"]): (float(r["mean_diff_mw"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in csv.DictReader(open(a.prefix + "_bootstrap.csv"))}
meta = json.load(open(a.prefix + "_meta.json"))
macros = {}

def g(tag, model, key, stat="mean"):
    r = rows.get((tag, model)); v = r.get(f"{key}_{stat}", "") if r else ""
    return float(v) if v not in ("", None) else np.nan
def pm(tag, model, key, d=1):
    m, s = g(tag, model, key), g(tag, model, key, "std")
    if np.isnan(m): return "--"
    return f"${m:.{d}f}$" if np.isnan(s) or s == 0 else f"${m:.{d}f} \\pm {s:.{d}f}$"
def ci(tag, other):
    b = boot.get((tag, other)); return "--" if b is None else f"${b[0]:+.1f}$ [{b[1]:.1f}, {b[2]:.1f}]"
def nseeds(tag, model): r = rows.get((tag, model)); return int(r["n_seeds"]) if r else 0
def mac(name, val): macros[name] = val

MODELS = ["PtO", "ICO", "ASYM", "OFFSET", "LS"]
LABEL = {"PtO": "PtO (MSE only)", "ICO": "ICO (QP layer)", "ASYM": "ASYM (closed form)", "OFFSET": "OFFSET (PtO $+ b$)", "LS": "LS (OLS)"}

# ---------------- main table (physical ceiling) ----------------
T = "base_a0.8_avphysical"
L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", "Metric & " + " & ".join(LABEL[m] for m in MODELS) + r" \\", r"\midrule",
     r"\multicolumn{6}{@{}l}{\textit{Forecast accuracy (test, MW unless stated)}}\\"]
for key, lab, d in [("test_mae", "MAE", 1), ("test_rmse", "RMSE", 1), ("test_r2", "$R^2$", 4), ("test_mape", "MAPE (\\%)", 3), ("test_bias", "Bias", 1)]:
    L.append(f"\\quad {lab} & " + " & ".join(pm(T, m, key, d) for m in MODELS) + r" \\")
L.append(r"\multicolumn{6}{@{}l}{\textit{Decision quality (test)}}\\")
for key, lab, d in [("test_shortfall_mw", "Mean modeled shortfall (MW)", 1), ("test_shortfall_rate", "Hours short (\\%)", 1),
                    ("test_excess_mw", "Mean excess commitment (MW)", 1), ("test_op_cost", "Operational cost (dimensionless)", 1), ("test_fossil_mw", "Fossil dispatch (MW)", 0)]:
    L.append(f"\\quad {lab} & " + " & ".join(pm(T, m, key, d) for m in MODELS) + r" \\")
L.append(r"\multicolumn{6}{@{}l}{\textit{Validation (selection set)}}\\")
for key, lab in [("val_mae", "MAE (MW)"), ("val_shortfall_mw", "Mean modeled shortfall (MW)")]:
    L.append(f"\\quad {lab} & " + " & ".join(pm(T, m, key, 1) for m in MODELS) + r" \\")
L.append(r"\multicolumn{6}{@{}l}{\textit{Computation}}\\")
L.append("\\quad Training time (min) & " + " & ".join(pm(T, m, "train_train_minutes", 1) if not np.isnan(g(T, m, "train_train_minutes")) else "$<0.1$" for m in MODELS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tab_main.tex"), "w").write("\n".join(L) + "\n")
for m in MODELS:
    for key, nm in [("test_mae", "MAE"), ("test_bias", "Bias"), ("test_shortfall_mw", "Short"), ("test_shortfall_rate", "Rate"), ("train_train_minutes", "Min"), ("val_mae", "ValMAE"), ("val_shortfall_mw", "ValShort"), ("test_fossil_mw", "Fossil"), ("test_clean_mw", "Clean")]:
        v = g(T, m, key)
        if not np.isnan(v): mac(f"{m}{nm}", f"{v:.1f}" if key != "train_train_minutes" else f"{v:.0f}")
    s = g(T, m, "test_shortfall_mw", "std");  mac(f"{m}ShortSd", f"{s:.1f}" if not np.isnan(s) else "")
    s = g(T, m, "test_bias", "std");  mac(f"{m}BiasSd", f"{s:.1f}" if not np.isnan(s) else "")
    s = g(T, m, "test_mae", "std");  mac(f"{m}MAESd", f"{s:.1f}" if not np.isnan(s) else "")
for m in MODELS:
    f, c = g(T, m, "test_fossil_mw"), g(T, m, "test_clean_mw")
    if not np.isnan(f): mac(f"{m}FossilPct", f"{100 * f / (f + c):.1f}")
mac("ShortRedPtO", f"{100 * (g(T, 'PtO', 'test_shortfall_mw') - g(T, 'ICO', 'test_shortfall_mw')) / g(T, 'PtO', 'test_shortfall_mw'):.1f}")
mac("ShortRedOffset", f"{100 * (g(T, 'OFFSET', 'test_shortfall_mw') - g(T, 'ICO', 'test_shortfall_mw')) / g(T, 'OFFSET', 'test_shortfall_mw'):.1f}")
mac("ShortRedLS", f"{100 * (g(T, 'LS', 'test_shortfall_mw') - g(T, 'ICO', 'test_shortfall_mw')) / g(T, 'LS', 'test_shortfall_mw'):.1f}")
mac("MAEIncPtO", f"{100 * (g(T, 'ICO', 'test_mae') - g(T, 'PtO', 'test_mae')) / g(T, 'PtO', 'test_mae'):.1f}")
mac("ValMAEIncPtO", f"{100 * (g(T, 'ICO', 'val_mae') - g(T, 'PtO', 'val_mae')) / g(T, 'PtO', 'val_mae'):.1f}")
mac("OffsetB", f"{g(T, 'OFFSET', 'train_offset_mw'):.0f}"); mac("KappaMean", f"{g(T, 'ICO', 'train_kappa'):,.0f}"); mac("KappaSd", f"{g(T, 'ICO', 'train_kappa', 'std'):,.0f}")
for o in ["PtO", "ASYM", "OFFSET"]:
    b = boot.get((T, o))
    if b: mac(f"CIvs{o}", f"${b[0]:+.1f}$ MW [{b[1]:.1f}, {b[2]:.1f}]")
dF = g(T, "ICO", "test_fossil_mw") - g(T, "PtO", "test_fossil_mw"); dC = g(T, "ICO", "test_clean_mw") - g(T, "PtO", "test_clean_mw")
dU = g(T, "PtO", "test_shortfall_mw") - g(T, "ICO", "test_shortfall_mw")
mac("BreakEven", f"{(57 * dF + 2 * dC) / dU:.0f}"); mac("AvoidedGWh", f"{dU * 8760 / 1000:.0f}"); mac("ExtraGenMW", f"{dF + dC:.0f}")
mac("SpeedUp", f"{g(T, 'ICO', 'train_train_minutes') / g(T, 'ASYM', 'train_train_minutes'):.0f}")

# ---------------- alpha table ----------------
alphas = sorted({float(t.split("_a")[1].split("_")[0]) for t, m in rows if t.startswith("base_a") and "testyear" not in t and "_hr" not in t and "_fl" not in t
                 and (t.endswith(SUF) if SUF else "_av" not in t) and m == "ICO"}, reverse=True)
def atag(al): return f"base_a{al:g}{SUF}"
L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
     r"$\alpha$ & val.\ MAE & val.\ shortfall & $\Delta$val.\ MAE vs PtO & test MAE & test bias & test shortfall & ICO $-$ ASYM \\",
     r" & (MW) & (MW) & (\%) & (MW) & (MW) & (MW) & (MW, 95\% CI) \\", r"\midrule"]
pv = g(T, "PtO", "val_mae")
L.append(f"1.0 (PtO) & {pm(T, 'PtO', 'val_mae')} & {pm(T, 'PtO', 'val_shortfall_mw')} & 0.0 & {pm(T, 'PtO', 'test_mae')} & {pm(T, 'PtO', 'test_bias')} & {pm(T, 'PtO', 'test_shortfall_mw')} & -- \\\\")
for al in alphas:
    t = atag(al)
    if (t, "ICO") not in rows: continue
    bold = al == 0.8
    row = f"{al:g} & {pm(t, 'ICO', 'val_mae')} & {pm(t, 'ICO', 'val_shortfall_mw')} & ${100 * (g(t, 'ICO', 'val_mae') - pv) / pv:+.1f}$ & {pm(t, 'ICO', 'test_mae')} & {pm(t, 'ICO', 'test_bias')} & {pm(t, 'ICO', 'test_shortfall_mw')} & {ci(t, 'ASYM')} \\\\"
    L.append((r"\rowcolor{gray!12} " if bold else "") + row)
    mac(f"AlphaValMAE{alname(al)}", f"{g(t, 'ICO', 'val_mae'):.1f}"); mac(f"AlphaValShort{alname(al)}", f"{g(t, 'ICO', 'val_shortfall_mw'):.1f}")
    mac(f"AlphaDValMAE{alname(al)}", f"{100 * (g(t, 'ICO', 'val_mae') - pv) / pv:+.1f}")
    mac(f"AlphaDValShort{alname(al)}", f"{100 * (g(T, 'PtO', 'val_shortfall_mw') - g(t, 'ICO', 'val_shortfall_mw')) / g(T, 'PtO', 'val_shortfall_mw'):.0f}")
    mac(f"AlphaTestMAE{alname(al)}", f"{g(t, 'ICO', 'test_mae'):.0f}"); mac(f"AlphaTestBias{alname(al)}", f"{g(t, 'ICO', 'test_bias'):+.0f}"); mac(f"AlphaTestShort{alname(al)}", f"{g(t, 'ICO', 'test_shortfall_mw'):.1f}")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tab_alpha.tex"), "w").write("\n".join(L) + "\n")

# ---------------- availability table ----------------
AV = [("Proxy (paper's rule)", "base_a0.8"), ("Physical (primary)", "base_a0.8_avphysical"), ("Realized (ex-post bound)", "base_a0.8_avrealized")]
L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
     r"Ceiling & \multicolumn{3}{c}{Mean modeled shortfall (MW)} & ICO $-$ OFFSET & Fossil share & Solar dispatch \\",
     r"\cmidrule(lr){2-4} & PtO & ICO & OFFSET & (MW, 95\% CI) & PtO / ICO (\%) & ICO (MW) \\", r"\midrule"]
for lab, t in AV:
    if (t, "ICO") not in rows: continue
    fp, cp = g(t, "PtO", "test_fossil_mw"), g(t, "PtO", "test_clean_mw"); fi, ci_ = g(t, "ICO", "test_fossil_mw"), g(t, "ICO", "test_clean_mw")
    sol = per_source(t, "ICO", "Solar")
    L.append(f"{lab} & {pm(t, 'PtO', 'test_shortfall_mw')} & {pm(t, 'ICO', 'test_shortfall_mw')} & {pm(t, 'OFFSET', 'test_shortfall_mw')} & {ci(t, 'OFFSET')} & "
             f"{100 * fp / (fp + cp):.1f} / {100 * fi / (fi + ci_):.1f} & {sol:.0f} \\\\")
    mac(f"Av{lab.split()[0]}SolarICO", f"{sol:,.0f}")
    key = lab.split()[0]
    mac(f"Av{key}ShortPtO", f"{g(t, 'PtO', 'test_shortfall_mw'):.1f}"); mac(f"Av{key}ShortICO", f"{g(t, 'ICO', 'test_shortfall_mw'):.1f}")
    mac(f"Av{key}Red", f"{100 * (g(t, 'PtO', 'test_shortfall_mw') - g(t, 'ICO', 'test_shortfall_mw')) / g(t, 'PtO', 'test_shortfall_mw'):.0f}")
    mac(f"Av{key}FossilICO", f"{100 * fi / (fi + ci_):.1f}"); mac(f"Av{key}BiasICO", f"{g(t, 'ICO', 'test_bias'):+.1f}")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tab_avail.tex"), "w").write("\n".join(L) + "\n")

# ---------------- conditions table ----------------
C = [("Base", f"base_a0.8{SUF}"), ("S2: demand volatility", f"s2_a0.8{SUF}"), ("S3: tight capacity", f"s3_a0.8{SUF}"),
     ("Rolling origin, test 2016", f"base_testyear2016_a0.8{SUF}"), ("Rolling origin, test 2017", f"base_testyear2017_a0.8{SUF}"), ("Rolling origin, test 2018", f"base_testyear2018_a0.8{SUF}")]
L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
     r"Condition & \multicolumn{3}{c}{Mean modeled shortfall (MW)} & \multicolumn{2}{c}{ICO $-$ other (MW, 95\% CI)} & \multicolumn{2}{c}{Bias (MW)} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8} & PtO & ICO & OFFSET & vs PtO & vs OFFSET & PtO & ICO \\", r"\midrule"]
for lab, t in C:
    if (t, "ICO") not in rows: continue
    L.append(f"{lab} & {pm(t, 'PtO', 'test_shortfall_mw')} & {pm(t, 'ICO', 'test_shortfall_mw')} & {pm(t, 'OFFSET', 'test_shortfall_mw')} & {ci(t, 'PtO')} & {ci(t, 'OFFSET')} & {pm(t, 'PtO', 'test_bias', 0)} & {pm(t, 'ICO', 'test_bias', 0)} \\\\")
    key = {"Base": "Base", "S2: demand volatility": "SII", "S3: tight capacity": "SIII"}.get(lab, "Y" + lab[-4:].replace("2016", "A").replace("2017", "B").replace("2018", "C"))
    mac(f"Cond{key}Red", f"{100 * (g(t, 'PtO', 'test_shortfall_mw') - g(t, 'ICO', 'test_shortfall_mw')) / g(t, 'PtO', 'test_shortfall_mw'):.0f}")
    b = boot.get((t, "OFFSET"));  mac(f"Cond{key}CIOffset", f"${b[0]:+.1f}$ MW [{b[1]:.1f}, {b[2]:.1f}]" if b else "--")
    b = boot.get((t, "ASYM"));  mac(f"Cond{key}CIAsym", f"${b[0]:+.1f}$ MW [{b[1]:.1f}, {b[2]:.1f}]" if b else "--")
    mac(f"Cond{key}BiasICO", f"{g(t, 'ICO', 'test_bias'):+.0f}"); mac(f"Cond{key}Slack", f"{g(t, 'ICO', 'test_slack_hours'):.0f}")
    mac(f"Cond{key}MAEICO", f"{g(t, 'ICO', 'test_mae'):.0f}"); mac(f"Cond{key}MAEPtO", f"{g(t, 'PtO', 'test_mae'):.0f}")
    s3 = meta.get(t, {}).get("s3")
    if s3: mac("SIIIScale", f"{s3['scale']:.3f}"); mac("SIIIBindTest", f"{100 * s3['bind_test']:.1f}"); mac("SIIIBindTrainVal", f"{100 * s3['bind_trainval']:.1f}")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tab_conditions.tex"), "w").write("\n".join(L) + "\n")

# ---------------- capacity grid ----------------
capvals = []
L = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule", r"Headroom & Floor (\%) & MAE (MW) & Bias (MW) & Shortfall (MW) & Op.\ cost \\", r"\midrule"]
for h in [1.1, 1.2, 1.3]:
    for f in [0.02, 0.05, 0.1]:
        t = "base_a0.8" + ("" if abs(h - 1.2) < 1e-9 else f"_hr{h:g}") + ("" if abs(f - 0.05) < 1e-9 else f"_fl{f:g}") + SUF
        if (t, "ICO") not in rows: continue
        L.append(f"{h:g} & {100 * f:g} & {g42(t, 'mae'):.1f} & {g42(t, 'bias'):+.1f} & {g42(t, 'shortfall_mw'):.1f} & {g42(t, 'op_cost'):.1f} \\\\")
        capvals.append((h, f, g42(t, 'shortfall_mw'), g42(t, 'bias')))
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tab_capgrid.tex"), "w").write("\n".join(L) + "\n")
if capvals:
    sh = [v[2] for v in capvals]; bi = [v[3] for v in capvals]
    mac("CapShortLo", f"{min(sh):.1f}"); mac("CapShortHi", f"{max(sh):.1f}"); mac("CapBiasLo", f"{min(bi):+.1f}"); mac("CapBiasHi", f"{max(bi):+.1f}")
    byh = {}
    for h, f, s_, b_ in capvals: byh.setdefault(h, []).append(s_)
    mac("CapFloorMaxEffect", f"{max(max(v) - min(v) for v in byh.values()):.1f}")
# largest |ICO - ASYM| mean shortfall difference over all tags of the chosen availability model
asym = [abs(v[0]) for (t, o), v in boot.items() if o == "ASYM" and (t.endswith(SUF) if SUF else "_av" not in t)]
mac("MaxAsymDiff", f"{max(asym):.1f}" if asym else "--")
mac("AlphaDValMAEEightyExact", f"{100 * (g('base_a0.8' + SUF, 'ICO', 'val_mae') - g(T, 'PtO', 'val_mae')) / g(T, 'PtO', 'val_mae'):.2f}")

# ---------------- split ----------------
mt = meta.get(T) or next(iter(meta.values()))
n, nt = mt["n_usable"], mt["n_test"]; ntr, nv = int(n * 0.8), int(n * 0.9) - int(n * 0.8)
L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Partition & Hours & Share \\", r"\midrule",
     f"Training & ${ntr:,}$ & $80\\%$ \\\\", f"Validation & ${nv:,}$ & $10\\%$ \\\\", f"Test & ${nt:,}$ & $10\\%$ \\\\", r"\midrule", f"Total & ${n:,}$ & $100\\%$ \\\\", r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tab_split.tex"), "w").write("\n".join(L) + "\n")
mac("NUsable", f"{n:,}"); mac("NTrain", f"{ntr:,}"); mac("NVal", f"{nv:,}"); mac("NTest", f"{nt:,}"); mac("TestStart", mt["test_start"][:16])

nruns = sum(1 for f in glob.glob(os.path.join(ROOT, "*", "seed*", "results.json")) if {"ICO", "ASYM"} <= set(json.load(open(f))["models"]))
mac("NRuns", str(nruns))
with open(os.path.join(a.out, "numbers.tex"), "w") as fh:
    fh.write("% auto-generated by make_tables.py; do not edit\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print("wrote", sorted(os.listdir(a.out)))
