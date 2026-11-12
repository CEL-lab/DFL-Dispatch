# Per-hour comparison of the QP layer (cvxpylayers/SCS) and the closed form of Proposition 1:
# backward values near the price plateau D_c and near total capacity C_bar.
# Uses the data, ceilings, layer and closed form of run_experiment.py unchanged (its sections 1-2).
# Usage:  SCENARIO=s3 AVAIL=proxy SAVE_DIR=/tmp/x python check_layer_vs_closed.py
import os, numpy as np
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment.py")).read()
exec(compile(_src.split("# self-check: closed form")[0], "run_experiment.py", "exec"))
import torch

rng = np.random.default_rng(3)
ix = rng.choice(np.arange(0, vl_end), size=200, replace=False)          # training + validation hours
capb = cap_gw[ix]; Cb = capb.sum(1); Dc = capb[:, CLEAN_IDX].sum(1)
Y = torch.tensor(y[ix] / 1000, dtype=torch.float64)

def backward(fn, D):
    D = D.clone().requires_grad_(True); v = fn(D, Y, capb); (g,) = torch.autograd.grad(v.sum(), D)
    return v.detach(), g

def served_slope(D):
    D = D.clone().requires_grad_(True)
    z, _ = qp_layer(D.reshape(-1, 1), capb, cost_t.expand(len(D), -1), solver_args=SOLVER_ARGS)
    (g,) = torch.autograd.grad(z.sum(1).sum(), D); return g

print(f"scenario={SCENARIO} avail={AVAIL} {s3_info}")
print("offset of forecast from C_bar | max |layer - closed| backward | share > 0.05 | median 1'dz/dh (exact: 1 below, 0 above)")
for off_mw in [-500, -50, -5, 5, 20, 50, 100, 200, 500, 2000]:
    D = Cb + off_mw / 1000
    _, gq = backward(oc_qp, D); _, gc = backward(oc_closed, D); d = (gq - gc).abs()
    print(f"  {off_mw:+6d} MW | {d.max():9.3f} | {float((d > 0.05).double().mean()):.3f} | {float(served_slope(D).median()):.3f}")
print("offset of forecast from D_c (price plateau) | max |layer - closed| backward | median 1'dz/dh")
for off_mw in [-10, -1, 0, 1, 10]:
    D = Dc + off_mw / 1000
    _, gq = backward(oc_qp, D); _, gc = backward(oc_closed, D)
    print(f"  {off_mw:+4d} MW | {(gq - gc).abs().max():9.3f} | {float(served_slope(D).median()):.3f}")
