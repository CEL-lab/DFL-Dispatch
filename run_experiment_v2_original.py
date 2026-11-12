#!/usr/bin/env python
# =====================================================================
# run_experiment.py  --  corrected runner for the EJOR revision (Stages A-B)
#
# One job = one seed x one scenario x one alpha. Trains and evaluates, on the
# SAME data, split, features and seed:
#   PtO    MSE-only network (identical architecture/hyperparameters)
#   ICO    composite loss, operational cost computed THROUGH the CvxpyLayer QP
#   ASYM   identical composite loss, but the dispatch cost is evaluated in
#          closed form WITHOUT an optimization layer (asymmetric-loss baseline, R7)
#   OFFSET PtO + one constant offset b, chosen on VALIDATION with the objective
#          ICO trains on, then frozen (calibrated baseline, R7)
#   LS     OLS on the same features (secondary benchmark)
# Every forecast is dispatched through the same per-hour QP (ECOS) on test.
#
# Fixes relative to ico_train_single_seed.py / three_model_comparison.py:
#   F1  rolling load statistics use load up to t-1 only (were including y_t)
#   F2  renewable capacity ceiling uses generation up to t-1 only (included t)
#   F3  series re-indexed to a regular hourly grid; rows whose lags cross a gap
#       are dropped (shift() previously skipped over 3 gaps)
#   F4  ICO trains with the hourly hybrid capacities of Eq. 6 (the base ICO run
#       used static maxima)
#   F5  best-checkpoint tracking is reset when the QP term switches on, so the
#       saved model is chosen on the composite objective, not on warm-up MSE
#   F6  QP failures are counted and skipped, never silently replaced by MSE;
#       the job aborts if more than MAX_QP_FAIL_FRAC of batches fail
#   F7  S3 capacity scale is calibrated on TRAIN+VALIDATION, not test (R8);
#       the binding rate actually achieved on test is reported
#   F8  validation predictions and metrics are saved for every model so alpha
#       and offsets are selected on validation only (R8)
#   F9  slack frequency, solver status counts, package versions, split
#       timestamps and timings are written to one results file per job (R10/R12)
#
# Env vars (defaults reproduce the base configuration):
#   SEED=42  DATA_DIR=.  SAVE_DIR=results/v2  ALPHA=0.80  MODELS=PtO+ICO+ASYM+LS
#   SCENARIO=base|s2|s3  NOISE_SIGMA=2000  NOISE_SEED=20240101  S3_BIND_FRAC=0.15
#   TEST_YEAR=  (e.g. 2017: train on prior years, test on that full year)
#   HEADROOM=1.2  FLOOR=0.05  MAX_EPOCHS=100  TRAIN_SUBSAMPLE=1 (testing only)
#   QP_SOLVER=  (empty = diffcp default, which is SCS; this is what the
#                original runs used)
#
# Example:  SEED=42 SCENARIO=base python run_experiment.py
# =====================================================================
import os, sys, time, json, platform, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cvxpy as cp
from torch.utils.data import DataLoader, TensorDataset
from cvxpylayers.torch import CvxpyLayer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

E = os.environ.get
SEED        = int(E("SEED", "42"))
DATA_DIR    = E("DATA_DIR", ".")
SCENARIO    = E("SCENARIO", "base").lower()
ALPHA       = float(E("ALPHA", "0.80"))
import re as _re   # accepts "PtO,ICO" or "PtO+ICO" (use + when passing through qsub -v, which splits on commas)
MODELS      = [m for m in _re.split(r"[,+ ]", E("MODELS", "PtO,ICO,ASYM,LS")) if m]
NOISE_SIGMA = float(E("NOISE_SIGMA", "2000"))
NOISE_SEED  = int(E("NOISE_SEED", "20240101"))
BIND_FRAC   = float(E("S3_BIND_FRAC", "0.15"))
TEST_YEAR   = E("TEST_YEAR", "").strip()
HEADROOM    = float(E("HEADROOM", "1.2"))
FLOOR       = float(E("FLOOR", "0.05"))
MAX_EPOCHS  = int(E("MAX_EPOCHS", "100"))
SUBSAMPLE   = int(E("TRAIN_SUBSAMPLE", "1"))
QP_SOLVER   = E("QP_SOLVER", "").strip()
MAX_QP_FAIL_FRAC = float(E("MAX_QP_FAIL_FRAC", "0.01"))
BATCH_SIZE, PATIENCE, WARMUP_EP = 64, 25, 5
LR, WEIGHT_DECAY = 5e-5, 2e-2
VOLL_OC, SLACK_PEN, SMOOTH = 1000.0, 1000.0, 0.1   # training weights, GW units
SHORT_TOL_MW = 1.0                                   # an hour counts as short if > 1 MW

tag = f"{SCENARIO}" + (f"_testyear{TEST_YEAR}" if TEST_YEAR else "") + f"_a{ALPHA:g}"
if abs(HEADROOM - 1.2) > 1e-9: tag += f"_hr{HEADROOM:g}"
if abs(FLOOR - 0.05) > 1e-9:   tag += f"_fl{FLOOR:g}"
SAVE_DIR = os.path.join(E("SAVE_DIR", "results/v2"), tag, f"seed{SEED}")
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[run] seed={SEED} scenario={SCENARIO} alpha={ALPHA} models={MODELS} -> {SAVE_DIR}", flush=True)

TARGET = "total load actual"
GEN_COLS = [
    "generation biomass", "generation fossil brown coal/lignite", "generation fossil gas",
    "generation fossil hard coal", "generation fossil oil",
    "generation hydro run-of-river and poundage", "generation hydro water reservoir",
    "generation nuclear", "generation other", "generation other renewable",
    "generation solar", "generation waste", "generation wind onshore"]
GEN_LABELS = ["Biomass", "Brown coal", "Gas", "Hard coal", "Oil", "Hydro RoR", "Hydro reservoir",
              "Nuclear", "Other", "Other renew.", "Solar", "Waste", "Wind"]
FOSSIL_COLS = GEN_COLS[1:5]
VARIABLE_RENEWABLE = ["generation solar", "generation wind onshore",
                      "generation hydro run-of-river and poundage", "generation hydro water reservoir"]
FOSSIL_IDX = [GEN_COLS.index(c) for c in FOSSIL_COLS]
CLEAN_IDX = [j for j in range(len(GEN_COLS)) if j not in FOSSIL_IDX]
cost_vec = np.array([10.0 if c in FOSSIL_COLS else 0.1 for c in GEN_COLS])
N_SRC = len(GEN_COLS)

# =====================================================================
# 1. DATA
# =====================================================================
df_e = pd.read_csv(os.path.join(DATA_DIR, "data", "modified_dataset.csv"))
if SCENARIO == "s2":   # same construction as scenario_s2.py
    noise = np.random.default_rng(NOISE_SEED).normal(0.0, NOISE_SIGMA, size=len(df_e))
    df_e[TARGET] = np.maximum(df_e[TARGET].to_numpy(float) + noise, 1.0)
df_e["time"] = pd.to_datetime(df_e["time"], utc=True)
df_w = pd.read_csv(os.path.join(DATA_DIR, "data", "weather_features.csv"))
df_w["time"] = pd.to_datetime(df_w["dt_iso"], utc=True)
df_w = df_w[["time", "temp", "humidity", "wind_speed", "clouds_all"]].groupby("time").mean().reset_index()
df = pd.merge(df_e, df_w, on="time", how="inner").dropna(subset=[TARGET] + GEN_COLS + ["temp"])

# F3: regular hourly grid so shift(k) means "k hours earlier"
df = df.set_index("time").sort_index()
n_raw = len(df)
df = df.asfreq("h").reset_index()
n_gap_hours = int(df[TARGET].isna().sum())

load = df[TARGET]
past = load.shift(1)                                   # F1: information up to t-1
for lag in [1, 2, 3, 24, 168]:
    df[f"load_lag_{lag}"] = load.shift(lag)
df["rolling_mean_24"]  = past.rolling(24).mean()
df["rolling_std_24"]   = past.rolling(24).std()
df["rolling_mean_168"] = past.rolling(168).mean()
df["rolling_max_24"]   = past.rolling(24).max()
df["hour_sin"]  = np.sin(2 * np.pi * df["time"].dt.hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["time"].dt.hour / 24)
df["month_sin"] = np.sin(2 * np.pi * df["time"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["time"].dt.month / 12)
df["day_of_week"] = df["time"].dt.dayofweek.astype(float)
df["is_weekend"]  = (df["time"].dt.dayofweek >= 5).astype(float)
df["temp_sq"]      = df["temp"] ** 2
df["temp_lag_24"]  = df["temp"].shift(24)
df["lag24_x_hour"] = df["load_lag_24"] * df["hour_sin"]
for col in VARIABLE_RENEWABLE:                          # F2: ceiling from generation up to t-1
    df["roll_" + col] = df[col].shift(1).rolling(24, min_periods=1).max()

FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend",
            "temp", "humidity", "wind_speed", "clouds_all", "temp_sq", "temp_lag_24",
            "load_lag_1", "load_lag_2", "load_lag_3", "load_lag_24", "load_lag_168",
            "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_max_24", "lag24_x_hour"]
df = df.dropna(subset=FEATURES + [TARGET] + GEN_COLS + ["roll_" + c for c in VARIABLE_RENEWABLE]).reset_index(drop=True)
n = len(df)

# ---- split ----
if TEST_YEAR:
    idx = np.where(df["time"].dt.year.values == int(TEST_YEAR))[0]
    assert len(idx) > 0 and idx[0] > 0, f"bad TEST_YEAR {TEST_YEAR}"
    vl_end, test_end = int(idx[0]), int(idx[-1] + 1)
    tr_end = int(vl_end * 0.9)
else:
    tr_end, vl_end, test_end = int(n * 0.8), int(n * 0.9), n
if SUBSAMPLE > 1:
    print(f"[run] TRAIN_SUBSAMPLE={SUBSAMPLE}: TESTING MODE, results not for the paper", flush=True)
split_info = {k: str(df["time"].iloc[i]) for k, i in
              [("train_start", 0), ("val_start", tr_end), ("test_start", vl_end), ("test_last", test_end - 1)]}
split_info.update(n_usable=n, n_train=tr_end, n_val=vl_end - tr_end, n_test=test_end - vl_end,
                  n_rows_after_merge=n_raw, n_gap_hours_inserted=n_gap_hours)
print(f"[data] {split_info}", flush=True)

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float64)
scaler = StandardScaler().fit(X[:tr_end])
mu_y, sig_y = float(y[:tr_end].mean()), float(y[:tr_end].std())
Xs = scaler.transform(X).astype(np.float32)
yn = ((y - mu_y) / sig_y).astype(np.float32)
sl_tr, sl_val, sl_te = slice(0, tr_end), slice(tr_end, vl_end), slice(vl_end, test_end)
y_val, y_test = y[sl_val], y[sl_te]

# ---- hybrid capacities (Eq. 6), MW ----
installed = df[GEN_COLS].iloc[:tr_end].max().values.astype(np.float64)
cap = np.tile(installed, (n, 1))
for col in VARIABLE_RENEWABLE:
    j = GEN_COLS.index(col)
    cap[:, j] = np.maximum(HEADROOM * df["roll_" + col].values, FLOOR * installed[j])

s3_info = {}
if SCENARIO == "s3":   # F7: calibrate on train+validation only
    ratio = y / cap.sum(1)
    scale = float(np.quantile(ratio[:vl_end], 1.0 - BIND_FRAC))
    cap *= scale
    installed *= scale
    s3_info = dict(scale=scale, bind_trainval=float(np.mean(ratio[:vl_end] > scale)),
                   bind_test=float(np.mean(ratio[sl_te] > scale)))
    print(f"[s3] {s3_info}", flush=True)

cap_gw = torch.tensor(cap / 1000.0, dtype=torch.float64)
cost_t = torch.tensor(cost_vec, dtype=torch.float64)
headroom_test = cap[sl_te].sum(1) - y_test
print(f"[data] min(total capacity - load) on test = {headroom_test.min():.0f} MW; "
      f"hours with load > capacity = {np.mean(headroom_test < 0):.3f}", flush=True)

# =====================================================================
# 2. DISPATCH: differentiable QP layer + exact closed form
# =====================================================================
z_v, s_v = cp.Variable(N_SRC), cp.Variable(1, nonneg=True)
h_p, cap_p, c_p = cp.Parameter(1), cp.Parameter(N_SRC, nonneg=True), cp.Parameter(N_SRC)
qp_layer = CvxpyLayer(
    cp.Problem(cp.Minimize(c_p @ z_v + SLACK_PEN * cp.sum(s_v) + SMOOTH * cp.sum_squares(z_v)),
               [z_v >= 0, z_v <= cap_p, cp.sum(z_v) + s_v == h_p]),
    parameters=[h_p, cap_p, c_p], variables=[z_v, s_v])
SOLVER_ARGS = {"solve_method": QP_SOLVER} if QP_SOLVER else {}

def _lambda(D, capb):
    """Marginal price lambda with sum_j clip((lambda-c_j)/(2*SMOOTH), 0, cap_j) = D (bisection)."""
    lo = torch.full_like(D, -1.0)
    hi = torch.full_like(D, float(cost_t.max()) + 2 * SMOOTH * float(capb.max()) + 1.0)
    for _ in range(60):
        m = (lo + hi) / 2
        z = torch.minimum(torch.clamp((m[:, None] - cost_t) / (2 * SMOOTH), min=0.0), capb).sum(1)
        lo, hi = torch.where(z < D, m, lo), torch.where(z < D, hi, m)
    return (lo + hi) / 2

class GenCost(torch.autograd.Function):
    """Optimal generation cost for total demand D (GW) under per-source caps; dG/dD = lambda."""
    @staticmethod
    def forward(ctx, D, capb):
        tot = capb.sum(1)
        Dc = torch.minimum(torch.clamp(D, min=0.0), tot)
        lam = _lambda(Dc, capb)
        z = torch.minimum(torch.clamp((lam[:, None] - cost_t) / (2 * SMOOTH), min=0.0), capb)
        ctx.save_for_backward(lam * ((D > 0) & (D < tot)).double())
        return (z * cost_t).sum(1) + SMOOTH * (z ** 2).sum(1)
    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return g * grad, None

def oc_closed(D_gw, y_gw, capb):
    """Operational cost of Eq. 13 without an optimization layer (exact for this QP)."""
    served = torch.minimum(D_gw, capb.sum(1))
    return GenCost.apply(D_gw, capb) + VOLL_OC * torch.relu(y_gw - served)

def oc_qp(D_gw, y_gw, capb):
    z, s = qp_layer(D_gw.reshape(-1, 1), capb, cost_t.expand(len(D_gw), -1), solver_args=SOLVER_ARGS)
    served = z.sum(1)
    return (z * cost_t).sum(1) + VOLL_OC * torch.relu(y_gw - served) + SMOOTH * (z ** 2).sum(1)

# self-check: closed form == QP layer (value and gradient) on random validation hours
_rng = np.random.default_rng(0)
_ix = _rng.choice(np.arange(tr_end, vl_end), size=64, replace=False)
_D = torch.tensor(y[_ix] / 1000 * _rng.uniform(0.9, 1.1, 64), dtype=torch.float64, requires_grad=True)
_Y = torch.tensor(y[_ix] / 1000, dtype=torch.float64)
_a = oc_qp(_D, _Y, cap_gw[_ix]); (_ga,) = torch.autograd.grad(_a.sum(), _D)
_b = oc_closed(_D, _Y, cap_gw[_ix]); (_gb,) = torch.autograd.grad(_b.sum(), _D)
selfcheck = dict(max_abs_value_diff=float((_a - _b).abs().max()), max_abs_grad_diff=float((_ga - _gb).abs().max()))
print(f"[check] QP layer vs closed form on 64 hours: {selfcheck}", flush=True)

# finite-difference gradient check of the QP layer away from kinks (R1 completion criterion)
def _fd_check(k=8, eps=1e-3):
    rows = []
    for i in _ix[:k]:
        d0 = y[i] / 1000 * 1.03
        f = lambda d: float(oc_qp(torch.tensor([d], dtype=torch.float64),
                                   torch.tensor([y[i] / 1000], dtype=torch.float64), cap_gw[[i]]).sum())
        Dt = torch.tensor([d0], dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(oc_qp(Dt, torch.tensor([y[i] / 1000], dtype=torch.float64), cap_gw[[i]]).sum(), Dt)
        rows.append((float(g), (f(d0 + eps) - f(d0 - eps)) / (2 * eps)))
    return rows
fd_rows = _fd_check()
print(f"[check] autodiff vs finite difference (dOC/dD at 3% over-forecast): {[(round(a,3),round(b,3)) for a,b in fd_rows]}", flush=True)

# =====================================================================
# 3. NETWORK AND TRAINING
# =====================================================================
class DemandForecaster(nn.Module):
    def __init__(self, d):
        super().__init__()
        L = []
        for a, b in [(d, 256), (256, 128), (128, 64)]:
            L += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(0.05)]
        self.net = nn.Sequential(*L, nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

to_gw = lambda p: torch.relu(p.double().cpu() * sig_y + mu_y) / 1000.0 + 1e-4

def train_mlp(mode):
    """mode: 'mse' (PtO) | 'qp' (ICO) | 'closed' (ASYM)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DemandForecaster(len(FEATURES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    tr_idx = np.arange(0, tr_end, SUBSAMPLE)
    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.tensor(Xs[tr_idx]), torch.tensor(yn[tr_idx]), torch.tensor(tr_idx)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g)
    Xv = torch.tensor(Xs[sl_val]).to(DEVICE); yv = torch.tensor(yn[sl_val]).to(DEVICE)
    yv_gw = torch.tensor(y_val / 1000.0); cap_v = cap_gw[sl_val]
    oc_fn = {"qp": oc_qp, "closed": oc_closed}.get(mode)
    kappa, best, best_w, best_ep, wait = None, float("inf"), None, None, 0
    n_batches = n_fail = 0
    hist = {k: [] for k in ["epoch", "train_mse", "train_oc", "val_mse", "val_oc", "val_loss"]}
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        if ep <= WARMUP_EP:
            for pg in opt.param_groups: pg["lr"] = LR * ep / WARMUP_EP
        dec = oc_fn is not None and ep > WARMUP_EP
        if dec and ep == WARMUP_EP + 1:                      # F5
            best, best_w, best_ep, wait = float("inf"), None, None, 0
        model.train(); mses, ocs = [], []
        for xb, yb, ib in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb); mse = torch.mean((pred - yb) ** 2)
            loss = mse
            if dec:
                n_batches += 1
                try:
                    oc = oc_fn(to_gw(pred), ((yb.double().cpu() * sig_y + mu_y) / 1000.0), cap_gw[ib]).mean()
                except Exception as e:                   # F6: count and skip, never fall back to MSE
                    n_fail += 1
                    if n_fail == 1: print(f"  [QP ERROR] {type(e).__name__}: {e}", flush=True)
                    if n_fail > max(5, MAX_QP_FAIL_FRAC * n_batches):
                        raise SystemExit(f"Aborting: {n_fail}/{n_batches} batches failed in the QP layer")
                    continue
                if kappa is None:
                    kappa = oc.item() / max(mse.item(), 1e-8)
                    print(f"  [kappa] OC={oc.item():.3f} MSE={mse.item():.5f} kappa={kappa:.1f}", flush=True)
                loss = ALPHA * mse + (1 - ALPHA) * oc.float().to(DEVICE) / kappa
                ocs.append(oc.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); mses.append(mse.item())
        if ep > WARMUP_EP: sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv); mv = torch.mean((pv - yv) ** 2).item()
            ocv = oc_closed(to_gw(pv), yv_gw, cap_v).mean().item()   # exact, identical for ICO and ASYM
        vloss = ALPHA * mv + (1 - ALPHA) * ocv / kappa if dec else mv
        for k, v in zip(hist, [ep, np.mean(mses), np.mean(ocs) if ocs else np.nan, mv, ocv, vloss]): hist[k].append(float(v))
        if vloss < best:
            best, best_ep, wait = vloss, ep, 0
            best_w = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE: break
        print(f"  [{mode}] ep {ep:3d}  train_mse {np.mean(mses):.5f}  val_mse {mv:.5f}  val_oc {ocv:.2f}  "
              f"val_loss {vloss:.6f}{' *' if best_ep == ep else ''}  {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best_w); model.eval()
    with torch.no_grad():
        pred = lambda sl: np.maximum(model(torch.tensor(Xs[sl]).to(DEVICE)).cpu().numpy() * sig_y + mu_y, 1.0)
        out = dict(val=pred(sl_val), test=pred(sl_te))
    info = dict(best_epoch=best_ep, epochs_run=ep, kappa=kappa, train_minutes=(time.time() - t0) / 60,
                qp_batches=n_batches, qp_failed_batches=n_fail, solver=(QP_SOLVER or "diffcp default (SCS)") if mode == "qp" else None)
    return out, info, hist, best_w

# =====================================================================
# 4. EVALUATION
# =====================================================================
def closed_metrics(p, yy, capm):
    """Validation-side metrics without a solver (exact for this QP)."""
    served = np.minimum(p, capm.sum(1)); short = np.maximum(0.0, yy - served)
    with torch.no_grad():
        oc = oc_closed(torch.tensor(p / 1000.0), torch.tensor(yy / 1000.0), torch.tensor(capm / 1000.0)).numpy()
    e = p - yy
    return dict(mae=float(np.abs(e).mean()), bias=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                mse_std=float(((e / sig_y) ** 2).mean()), shortfall_mw=float(short.mean()),
                shortfall_rate=float((short > SHORT_TOL_MW).mean() * 100), op_cost=float(oc.mean()))

def qp_test_eval(p):
    zv, sv = cp.Variable(N_SRC), cp.Variable(1)
    hv, cv = cp.Parameter(1), cp.Parameter(N_SRC)
    prob = cp.Problem(cp.Minimize(cost_vec @ zv + SLACK_PEN * cp.sum(sv) + SMOOTH * cp.sum_squares(zv)),
                      [zv >= 0, zv <= cv, sv >= 0, cp.sum(zv) + sv == hv])
    N = len(p); D = np.zeros((N, N_SRC)); slack = np.zeros(N); objs = np.zeros(N); status = {}
    for i in range(N):
        hv.value = np.array([max(p[i], 0.1) / 1000]); cv.value = np.maximum(cap[vl_end + i], 0) / 1000
        prob.solve(solver=cp.ECOS)
        status[prob.status] = status.get(prob.status, 0) + 1
        if prob.status in ("optimal", "optimal_inaccurate"):
            D[i] = zv.value * 1000; slack[i] = max(float(sv.value[0]) * 1000, 0.0); objs[i] = prob.value
        else:
            D[i] = 0.0; slack[i] = p[i]; objs[i] = np.nan
    served = D.sum(1); short = np.maximum(0.0, y_test - served); zg = D / 1000
    oc = zg @ cost_vec + VOLL_OC * short / 1000 + SMOOTH * (zg ** 2).sum(1)
    e = p - y_test
    m = dict(r2=float(r2_score(y_test, p)), mae=float(mean_absolute_error(y_test, p)),
             rmse=float(np.sqrt(mean_squared_error(y_test, p))), mape=float(np.mean(np.abs(e) / y_test) * 100),
             bias=float(e.mean()), shortfall_mw=float(short.mean()), shortfall_max_mw=float(short.max()),
             shortfall_rate=float((short > SHORT_TOL_MW).mean() * 100), shortfall_mwh=float(short.sum()),
             excess_mw=float(np.maximum(0.0, served - y_test).mean()),
             slack_hours=int((slack > SHORT_TOL_MW).sum()), identity_residual_mw=float(short.mean() - (np.abs(e).mean() - e.mean()) / 2),
             op_cost=float(oc.mean()), qp_obj=float(np.nanmean(objs)), qp_status=status,
             fossil_mw=float(D[:, FOSSIL_IDX].sum(1).mean()), clean_mw=float(D[:, CLEAN_IDX].sum(1).mean()),
             per_source_mw=dict(zip(GEN_LABELS, D.mean(0).round(2).tolist())))
    return m, D, short, slack

# =====================================================================
# 5. RUN
# =====================================================================
results = dict(config=dict(seed=SEED, scenario=SCENARIO, alpha=ALPHA, test_year=TEST_YEAR, headroom=HEADROOM,
                           floor=FLOOR, max_epochs=MAX_EPOCHS, train_subsample=SUBSAMPLE, noise_sigma=NOISE_SIGMA if SCENARIO == "s2" else None,
                           noise_seed=NOISE_SEED if SCENARIO == "s2" else None, s3=s3_info or None,
                           qp_solver_training=QP_SOLVER or "diffcp default (SCS)", qp_solver_eval="ECOS",
                           units="QP in GW: cost 10 (fossil) / 0.1 (clean) per GW, slack 1000 per GW, smoothing 0.1*sum z^2 (z in GW)"),
               split=split_info, checks=dict(qp_vs_closed=selfcheck, fd_grad=fd_rows),
               env=dict(python=platform.python_version(), torch=torch.__version__, cvxpy=cp.__version__,
                        numpy=np.__version__, pandas=pd.__version__, host=platform.node(), cpu=platform.processor(),
                        cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        threads=torch.get_num_threads()),
               models={})
try:
    import cvxpylayers, diffcp, scs, ecos, sklearn, scipy
    results["env"].update(cvxpylayers=cvxpylayers.__version__, diffcp=getattr(diffcp, "__version__", "?"),
                          scs=scs.__version__, ecos=ecos.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
except Exception:
    pass

preds, arrays = {}, {}
for name, mode in [("PtO", "mse"), ("ICO", "qp"), ("ASYM", "closed")]:
    if name not in MODELS and not (name == "PtO" and "OFFSET" in MODELS): continue
    print(f"\n===== {name} =====", flush=True)
    out, info, hist, w = train_mlp(mode)
    preds[name] = out; results["models"][name] = dict(train=info)
    np.save(os.path.join(SAVE_DIR, f"history_{name}.npy"), hist, allow_pickle=True)
    torch.save(w, os.path.join(SAVE_DIR, f"weights_{name}.pt"))

if "LS" in MODELS:
    A = np.column_stack([np.ones(tr_end), Xs[:tr_end]])
    beta, *_ = np.linalg.lstsq(A, y[:tr_end], rcond=None)
    f = lambda sl: np.maximum(np.column_stack([np.ones(sl.stop - sl.start), Xs[sl]]) @ beta, 1.0)
    preds["LS"] = dict(val=f(sl_val), test=f(sl_te)); results["models"]["LS"] = dict(train={})

if "PtO" in preds:   # OFFSET baseline, always computed when PtO is trained (R7)
    pv = preds["PtO"]["val"]; cv = cap[sl_val]
    m0 = closed_metrics(pv, y_val, cv); kv = m0["op_cost"] / m0["mse_std"]
    grid = np.arange(-500.0, 1000.1, 1.0)
    J = []
    for b in grid:
        mb = closed_metrics(pv + b, y_val, cv); J.append(ALPHA * mb["mse_std"] + (1 - ALPHA) * mb["op_cost"] / kv)
    b_star = float(grid[int(np.argmin(J))])
    preds["OFFSET"] = dict(val=pv + b_star, test=preds["PtO"]["test"] + b_star)
    results["models"]["OFFSET"] = dict(train=dict(offset_mw=b_star, kappa_val=kv,
        rule="b minimises alpha*MSE_std + (1-alpha)*OC/kappa on validation; kappa = OC/MSE_std of unshifted PtO on validation"))

for name, p in preds.items():
    print(f"[eval] {name}", flush=True)
    results["models"][name]["val"] = closed_metrics(p["val"], y_val, cap[sl_val])
    mt, D, short, slack = qp_test_eval(p["test"])
    results["models"][name]["test"] = mt
    arrays.update({f"{name}_val_pred": p["val"], f"{name}_test_pred": p["test"],
                   f"{name}_test_dispatch": D, f"{name}_test_shortfall": short, f"{name}_test_slack": slack})
arrays.update(y_val=y_val, y_test=y_test, time_test=df["time"].iloc[sl_te].astype(str).values, cap_test=cap[sl_te])
np.savez_compressed(os.path.join(SAVE_DIR, "hourly_outputs.npz"), **arrays)
with open(os.path.join(SAVE_DIR, "results.json"), "w") as fh:
    json.dump(results, fh, indent=1, default=float)

print("\n" + "=" * 78)
print(f"{'model':8s} {'val MAE':>8s} {'val short':>9s} | {'MAE':>7s} {'bias':>7s} {'short':>7s} {'rate%':>6s} {'slack h':>7s} {'best ep':>7s}")
for name, r in results["models"].items():
    v, t = r["val"], r["test"]
    print(f"{name:8s} {v['mae']:8.1f} {v['shortfall_mw']:9.1f} | {t['mae']:7.1f} {t['bias']:+7.1f} {t['shortfall_mw']:7.1f} "
          f"{t['shortfall_rate']:6.1f} {t['slack_hours']:7d} {str(r['train'].get('best_epoch', '-')):>7s}")
print(f"\nSaved -> {SAVE_DIR}/results.json, hourly_outputs.npz")
