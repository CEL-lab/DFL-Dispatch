#!/usr/bin/env python
# =====================================================================
# ICO Model — single-seed training (decision-focused, OC settled
# against the ACTUAL load). Reads SEED, DATA_DIR, SAVE_DIR from the
# environment (set by run_ico.pbs). Writes metrics_seed<SEED>.csv plus
# per-seed prediction/dispatch/weight files to SAVE_DIR.
#
# Run one seed:   SEED=42 python ico_train_single_seed.py
# On the cluster: qsub -v SEED=42 run_ico.pbs
# =====================================================================

# ============================================================
# CELL 1 — DATA LOADING AND FEATURE ENGINEERING
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cvxpy as cp
import time
import os
import warnings
from torch.utils.data import DataLoader, TensorDataset
from cvxpylayers.torch import CvxpyLayer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

# ── Paths and single-seed config from environment (set by PBS) ─
DATA_DIR = os.environ.get("DATA_DIR", ".")
SAVE_DIR = os.environ.get("SAVE_DIR", "results/ico_model")
SEED     = int(os.environ.get("SEED", "42"))
ENERGY_PATH  = os.path.join(DATA_DIR, "data", "modified_dataset.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "data", "weather_features.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────
ALPHA      = float(os.environ.get("ALPHA", "0.80"))   # composite loss weight; env-configurable for the alpha sweep
MAX_EPOCHS = 100
BATCH_SIZE = 64          # every sample in batch goes through QP
PATIENCE   = 25
WARMUP_EP  = 5           # MSE-only warmup before QP activates
LR         = 5e-5
WEIGHT_DECAY = 2e-2

# ── Sensitivity / sampling controls (all default to the base configuration) ──
# These let the reviewer-requested experiments run WITHOUT editing anything else:
#   HEADROOM / FLOOR : renewable capacity constants (base 1.2 and 0.05). Vary them
#                      for the capacity-constant sensitivity sweep (Dr. Le, pt. 2).
#   YEAR             : per-year evaluation (Dr. Le, pt. 4). If set (e.g. "2015"),
#                      the data is restricted to that calendar year and split
#                      chronologically 80/10/10 within the year. Run 2015..2018 as
#                      four self-contained experiments. Chronology is preserved:
#                      train (early year) -> val (mid) -> test (late year).
#   TEST_YEAR        : train-past / test-next-FULL-year (Dr. Le, pt. 4, alt). If set
#                      (e.g. "2018"), train on all prior years, validate on the tail
#                      of that period, and test on every hour of TEST_YEAR (all
#                      seasons). Run 2016, 2017, 2018 as an expanding-by-year study;
#                      the 2018 fold is close to the paper's headline split.
#   N_FOLDS / FOLD   : rolling-origin walk-forward CV (another alternative). Also
#                      temporally valid; N_FOLDS=1 reproduces the original split.
# Outputs auto-route to a per-configuration sub-folder so runs never overwrite each
# other. With no env vars set, behaviour AND output paths match the original exactly.
HEADROOM  = float(os.environ.get("HEADROOM", "1.2"))
FLOOR     = float(os.environ.get("FLOOR",    "0.05"))
YEAR      = os.environ.get("YEAR", "").strip()
TEST_YEAR = os.environ.get("TEST_YEAR", "").strip()
N_FOLDS   = int(os.environ.get("N_FOLDS", "1"))
FOLD      = int(os.environ.get("FOLD",    "0"))
if YEAR and TEST_YEAR:
    raise SystemExit("Set only ONE of YEAR (per-year) or TEST_YEAR (train-past/test-next).")
if YEAR or TEST_YEAR:         # both force a single custom split (not rolling folds)
    N_FOLDS, FOLD = 1, 0

_tag = os.environ.get("RUN_TAG", "")
if not _tag:
    _parts = []
    if YEAR:                       _parts.append(f"year{YEAR}")
    if TEST_YEAR:                  _parts.append(f"testyear{TEST_YEAR}")
    if N_FOLDS != 1:               _parts.append(f"fold{FOLD}of{N_FOLDS}")
    if abs(HEADROOM - 1.2) > 1e-9: _parts.append(f"hr{HEADROOM:g}")
    if abs(FLOOR - 0.05) > 1e-9:   _parts.append(f"fl{FLOOR:g}")
    if abs(ALPHA - 0.80) > 1e-9:   _parts.append(f"alpha{ALPHA:g}")
    _tag = "_".join(_parts)
if _tag:
    SAVE_DIR = os.path.join(SAVE_DIR, _tag)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Run tag: {_tag}  ->  outputs in {SAVE_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Single-seed run | SEED = {SEED}")

# ── Column definitions ───────────────────────────────────────
TARGET = "total load actual"

GEN_COLS = [
    "generation biomass",
    "generation fossil brown coal/lignite",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation fossil oil",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
    "generation nuclear",
    "generation other",
    "generation other renewable",
    "generation solar",
    "generation waste",
    "generation wind onshore",
]

FOSSIL_COLS = [
    "generation fossil brown coal/lignite",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation fossil oil",
]

VARIABLE_RENEWABLE = [
    "generation solar",
    "generation wind onshore",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
]

GEN_LABELS = [
    "Biomass", "Coal", "Gas", "Hard Coal", "Oil",
    "Hydro ROR", "Hydro Res", "Nuclear", "Other",
    "Oth.Renew", "Solar", "Waste", "Wind",
]

FOSSIL_LABELS = ["Coal", "Gas", "Hard Coal", "Oil"]

# ── Load and merge data ──────────────────────────────────────
df_energy = pd.read_csv(ENERGY_PATH)
df_energy["time"] = pd.to_datetime(df_energy["time"], utc=True)

df_weather = pd.read_csv(WEATHER_PATH)
df_weather["time"] = pd.to_datetime(df_weather["dt_iso"], utc=True)
df_weather = (
    df_weather[["time", "temp", "humidity", "wind_speed", "clouds_all"]]
    .groupby("time").mean().reset_index()
)

df = pd.merge(df_energy, df_weather, on="time", how="inner")
df = df.dropna(subset=[TARGET] + GEN_COLS + ["temp"])

# ── Feature engineering (22 features) ────────────────────────
for lag in [1, 2, 3, 24, 168]:
    df[f"load_lag_{lag}"] = df[TARGET].shift(lag)

df["rolling_mean_24"]  = df[TARGET].rolling(24).mean()
df["rolling_std_24"]   = df[TARGET].rolling(24).std()
df["rolling_mean_168"] = df[TARGET].rolling(168).mean()
df["rolling_max_24"]   = df[TARGET].rolling(24).max()

df["hour_sin"]  = np.sin(2 * np.pi * df["time"].dt.hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["time"].dt.hour / 24)
df["month_sin"] = np.sin(2 * np.pi * df["time"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["time"].dt.month / 12)

df["day_of_week"] = df["time"].dt.dayofweek.astype(float)
df["is_weekend"]  = (df["time"].dt.dayofweek >= 5).astype(float)

df["temp_sq"]     = df["temp"] ** 2
df["temp_lag_24"] = df["temp"].shift(24)
df["lag24_x_hour"] = df["load_lag_24"] * df["hour_sin"]

df = df.dropna().reset_index(drop=True)

FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",
    "temp", "humidity", "wind_speed", "clouds_all",
    "temp_sq", "temp_lag_24",
    "load_lag_1", "load_lag_2", "load_lag_3", "load_lag_24", "load_lag_168",
    "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_max_24",
    "lag24_x_hour",
]

# ── Train / Validation / Test split ──────────────────────────
# N_FOLDS == 1 : original single 80/10/10 chronological split.
# N_FOLDS  > 1 : rolling-origin (expanding-window) cross-validation. Training
#   always precedes validation, which always precedes test, so no future
#   information leaks through the lag/rolling features. Val and test blocks are
#   10% of the series each and march forward as FOLD increases; the LAST fold
#   (FOLD = N_FOLDS-1) reproduces the original 80/10/10 test block. This is the
#   temporally valid analogue of k-fold — naive k-fold would shuffle time and
#   leak the future backward through the lags, so it is deliberately NOT used.
n = len(df)
if YEAR:
    # Restrict to one calendar year. The lag/rolling columns were already computed
    # on the full continuous series (and NaNs dropped above), so early-January rows
    # keep valid lags that reference the previous December. Splitting happens within
    # the year, so chronology and causality are preserved.
    df = df[df["time"].dt.year == int(YEAR)].reset_index(drop=True)
    n = len(df)
    print(f"Per-year mode: calendar year {YEAR} only ({n} usable hours)")

if TEST_YEAR:
    # Train on ALL prior years, validate on the tail (last 10%) of that prior period,
    # and test on every hour of TEST_YEAR. Full series is kept; rows after TEST_YEAR
    # are simply unused (never trained on). Chronology: train -> val -> test.
    _years = df["time"].dt.year.values
    _idx = np.where(_years == int(TEST_YEAR))[0]
    assert len(_idx) > 0, f"No rows found for TEST_YEAR={TEST_YEAR}"
    test_start, test_end = int(_idx[0]), int(_idx[-1] + 1)
    assert test_start > 0, f"TEST_YEAR={TEST_YEAR} has no prior years to train on"
    vl_end = test_start
    tr_end = int(test_start * 0.9)              # last 10% of prior period = validation
    print(f"Test-next-year mode: train+val on years < {TEST_YEAR}, test on all "
          f"{TEST_YEAR} ({test_end - test_start} test hrs; {test_start} prior hrs)")
elif N_FOLDS == 1:
    tr_end   = int(n * 0.8)
    vl_end   = int(n * 0.9)
    test_end = n
else:
    assert 0 <= FOLD < N_FOLDS, "FOLD must be in [0, N_FOLDS)"
    block = int(n * 0.10)                       # val and test blocks: 10% each
    assert n - (N_FOLDS + 1) * block > int(n * 0.20), \
        "Too many folds: the earliest fold's training window would be < 20% of the series"
    test_end = n - (N_FOLDS - 1 - FOLD) * block
    vl_end   = test_end - block                 # test = [vl_end, test_end)
    tr_end   = vl_end - block                    # val  = [tr_end, vl_end)
print(f"Split: train[0:{tr_end}] val[{tr_end}:{vl_end}] test[{vl_end}:{test_end}]  "
      f"(N_FOLDS={N_FOLDS}, FOLD={FOLD})")

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

scaler = StandardScaler().fit(X[:tr_end])
mu_y  = float(y[:tr_end].mean())
sig_y = float(y[:tr_end].std())

X_scaled = scaler.transform(X)
y_norm   = (y - mu_y) / sig_y

X_train = torch.FloatTensor(X_scaled[:tr_end])
y_train = torch.FloatTensor(y_norm[:tr_end])
X_val   = torch.FloatTensor(X_scaled[tr_end:vl_end])
y_val   = torch.FloatTensor(y_norm[tr_end:vl_end])
X_test  = torch.FloatTensor(X_scaled[vl_end:test_end])
y_test_tensor = torch.FloatTensor(y_norm[vl_end:test_end])   # standardized, for test loss
y_test  = y[vl_end:test_end]                                  # original scale, for metrics

# ── Generator capacities and cost vector ─────────────────────
n_sources = len(GEN_COLS)
installed_cap = df[GEN_COLS].iloc[:tr_end].max().values.astype(np.float32)

# Hybrid capacity model. Variable renewables get an hour-by-hour ceiling equal to
# HEADROOM x (trailing-24h rolling max), floored at FLOOR x installed capacity.
# HEADROOM (base 1.2) and FLOOR (base 0.05) come from the environment so the
# capacity-constant sensitivity sweep runs without touching anything else.
capacities = np.zeros((n, n_sources), dtype=np.float32)
for j, col in enumerate(GEN_COLS):
    if col in VARIABLE_RENEWABLE:
        capacities[:, j] = np.maximum(
            df[col].rolling(24, min_periods=1).max().values * HEADROOM,
            FLOOR * installed_cap[j]
        ).astype(np.float32)
    else:
        capacities[:, j] = installed_cap[j]

cap_test = capacities[vl_end:test_end]

# Per-hour capacity slices (GW) for the embedded QP. Training uses the true
# hour-by-hour hybrid ceilings (Eq. 6), so HEADROOM/FLOOR actually shape training.
cap_train_hourly = torch.DoubleTensor(capacities[:tr_end].astype(np.float64) / 1000.0)
cap_val_hourly   = torch.DoubleTensor(capacities[tr_end:vl_end].astype(np.float64) / 1000.0)
cap_test_hourly  = torch.DoubleTensor(capacities[vl_end:test_end].astype(np.float64) / 1000.0)

# DIAGNOSTIC: print the actual renewable ceilings so each log PROVES whether
# HEADROOM/FLOOR took effect (a fixed run shows these change with the constants;
# a stale/cached run shows the default values regardless of the banner).
_wind_j = GEN_COLS.index("generation wind onshore") if "generation wind onshore" in GEN_COLS else 0
_solar_j = GEN_COLS.index("generation solar") if "generation solar" in GEN_COLS else 0
print(f"[CAP-CHECK] HEADROOM={HEADROOM} FLOOR={FLOOR} | "
      f"wind mean ceiling={capacities[:, _wind_j].mean():.1f} MW | "
      f"solar mean ceiling={capacities[:, _solar_j].mean():.1f} MW | "
      f"wind floor={FLOOR*installed_cap[_wind_j]:.1f} MW", flush=True)

cost_vec = np.array(
    [10.0 if c in FOSSIL_COLS else 0.1 for c in GEN_COLS],
    dtype=np.float32,
)

n_batches = int(np.ceil(tr_end / BATCH_SIZE))
print(f"Data: {n:,} hrs | Train {tr_end:,} | Val {vl_end - tr_end:,} | Test {test_end - vl_end:,}")
print(f"Load: {mu_y:,.0f} ± {sig_y:,.0f} MW | {n_sources} sources | {len(FEATURES)} features")
print(f"ICO training: {n_batches} batches × {BATCH_SIZE} samples = {n_batches * BATCH_SIZE:,} QP solves/epoch")

# ============================================================
# CELL 2 — DIFFERENTIABLE QP DISPATCH LAYER
# ============================================================
# The QP is wrapped in CvxpyLayer, which computes dz*/dy_hat
# through the KKT conditions using implicit differentiation.
# ============================================================

z_qp   = cp.Variable(n_sources)
s_qp   = cp.Variable(1, nonneg=True)
h_qp   = cp.Parameter(1)
cap_qp = cp.Parameter(n_sources, nonneg=True)
c_qp   = cp.Parameter(n_sources)

qp_objective = cp.Minimize(
    c_qp @ z_qp + 1000 * cp.sum(s_qp) + 0.1 * cp.sum_squares(z_qp)
)
qp_constraints = [
    z_qp >= 0,
    z_qp <= cap_qp,
    cp.sum(z_qp) + s_qp == h_qp,
]

qp_layer = CvxpyLayer(
    cp.Problem(qp_objective, qp_constraints),
    parameters=[h_qp, cap_qp, c_qp],
    variables=[z_qp, s_qp],
)

cap_tensor  = torch.FloatTensor(installed_cap / 1000).to(DEVICE)
cost_tensor = torch.FloatTensor(cost_vec).to(DEVICE)
# double-precision copies for the CvxpyLayer/diffcp solve (avoids float/double dtype clash)
cap_tensor_d  = cap_tensor.double()
cost_tensor_d = cost_tensor.double()
# This cvxpylayers build cannot consume CUDA tensors, so the QP always runs on CPU.
# We pass CPU copies and move only the scalar cost back to the model's device.
cap_cpu  = cap_tensor_d.cpu()
cost_cpu = cost_tensor_d.cpu()

# ── Gradient verification ────────────────────────────────────
# Operational-cost penalty on unserved energy (relative training weight, not the euro VoLL).
VOLL_OC = 1000.0

# The operational cost is settled against the ACTUAL load: the dispatch is committed to
# meet the forecast, then any shortfall versus the realised demand is charged at VOLL_OC.
# Under-prediction -> negative gradient (push forecast up); over-prediction -> mild pull down.
y_actual = 30.0  # realised load [GW] for the check

h_under = torch.tensor([28.0], dtype=torch.float64, requires_grad=True)   # forecast below actual
z_u, s_u = qp_layer(h_under, cap_tensor_d.cpu(), cost_tensor_d.cpu())
oc_u = cost_tensor_d.cpu() @ z_u + VOLL_OC * torch.relu(torch.tensor(y_actual, dtype=torch.float64) - z_u.sum()) + 0.1 * (z_u ** 2).sum()
oc_u.backward()
print(f"Gradient check (under-forecast): dOC/dy_hat = {h_under.grad.item():.2f}  "
      f"(y_hat=28, actual=30 GW; negative => pushes forecast up)")

h_over = torch.tensor([32.0], dtype=torch.float64, requires_grad=True)    # forecast above actual
z_o, s_o = qp_layer(h_over, cap_tensor_d.cpu(), cost_tensor_d.cpu())
oc_o = cost_tensor_d.cpu() @ z_o + VOLL_OC * torch.relu(torch.tensor(y_actual, dtype=torch.float64) - z_o.sum()) + 0.1 * (z_o ** 2).sum()
oc_o.backward()
print(f"Gradient check (over-forecast):  dOC/dy_hat = {h_over.grad.item():.2f}  "
      f"(y_hat=32, actual=30 GW; small positive => mild pull down)")

# ============================================================
# CELL 3 — NEURAL NETWORK ARCHITECTURE
# ============================================================

class DemandForecaster(nn.Module):
    """
    Feedforward neural network for hourly demand forecasting.
    Architecture: 22 → 256 → 128 → 64 → 1
    Regularization: LayerNorm + GELU + Dropout(0.05)
    Parameters: ~47,000
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def prediction_to_gw(pred_normalized):
    """Convert normalized NN output to demand in GW (for QP input)."""
    return torch.relu(pred_normalized * sig_y + mu_y) / 1000.0 + 1e-4


# Count parameters
_tmp = DemandForecaster(len(FEATURES))
n_params = sum(p.numel() for p in _tmp.parameters())
print(f"Architecture: MLP {len(FEATURES)}→256→128→64→1")
print(f"Total parameters: {n_params:,}")
del _tmp

# ============================================================
# CELL 4 — ICO MODEL TRAINING (MULTI-SEED)
# ============================================================
# Trains across multiple seeds for statistical robustness.
# Each seed: full training → test prediction → QP dispatch
# ============================================================

def train_ico_single_seed(seed):
    """Train one ICO model with the given seed. Returns all results."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DemandForecaster(len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Reproducible DataLoader shuffle. idx_train carries each sample's row index
    # so the QP uses that hour's capacity ceiling (cap_train_hourly[idx]).
    idx_train = torch.arange(X_train.shape[0])
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, idx_train), batch_size=BATCH_SIZE,
        shuffle=True, generator=g
    )

    history = {k: [] for k in ["epoch", "train_loss", "val_loss",
                                "mse_train", "oc_train", "mse_val", "oc_val"]}
    best_val_loss = float("inf")
    best_weights  = None
    wait          = 0
    oc_scale      = None
    gradient_norms = []
    qp_fail_count = 0

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'MSE(train)':>12}  "
          f"{'OC(train)':>12}  {'MSE(val)':>12}  {'Status':>8}  {'Time':>6}")
    print("-" * 96)

    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()

        if epoch <= WARMUP_EP:
            for pg in optimizer.param_groups:
                pg["lr"] = LR * epoch / WARMUP_EP

        model.train()
        mse_batch, oc_batch, gnorm_batch = [], [], []
        qp_active = epoch > WARMUP_EP

        for xb, yb, ib in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            pred = model(xb).squeeze()
            mse = torch.mean((pred - yb) ** 2)

            if qp_active:
                pred_gw = prediction_to_gw(pred)                 # [B] on DEVICE
                Bk = pred_gw.shape[0]
                try:
                    h_b = pred_gw.double().cpu().reshape(Bk, 1)          # [B,1] CPU double
                    z_b, s_b = qp_layer(h_b,
                                        cap_train_hourly[ib],
                                        cost_cpu.unsqueeze(0).expand(Bk, -1))   # z_b [B,n] CPU
                    served    = z_b.sum(dim=1)                                  # [B]
                    y_gw_b    = ((yb * sig_y + mu_y) / 1000.0).double().cpu()   # [B] actual load
                    shortfall = torch.relu(y_gw_b - served)                     # unserved vs actual
                    fuel      = (z_b * cost_cpu.unsqueeze(0)).sum(dim=1)         # [B]
                    smooth    = 0.1 * (z_b ** 2).sum(dim=1)                     # [B]
                    oc        = (fuel + VOLL_OC * shortfall + smooth).mean()    # scalar CPU double
                except Exception as e:
                    if qp_fail_count == 0:
                        print(f"  [QP ERROR] {type(e).__name__}: {e}")
                    qp_fail_count += Bk
                    oc = None

                if oc is not None:
                    if oc_scale is None:
                        oc_scale = oc.item() / max(mse.item(), 1e-8)
                        print(f"  [AUTO-SCALE] OC={oc.item():.2f}  MSE={mse.item():.4f}  "
                              f"ratio={oc_scale:.1f}")
                    total_loss = ALPHA * mse + (1 - ALPHA) * oc.float().to(DEVICE) / oc_scale
                    oc_batch.append(oc.item())
                else:
                    total_loss = mse
                    oc_batch.append(0.0)
            else:
                total_loss = mse
                oc_batch.append(0.0)

            total_loss.backward()

            g_norm = model.net[0].weight.grad
            if g_norm is not None:
                gnorm_batch.append(g_norm.norm().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mse_batch.append(mse.item())

        if epoch > WARMUP_EP:
            scheduler.step()

        gradient_norms.append(np.mean(gnorm_batch) if gnorm_batch else 0.0)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val.to(DEVICE)).squeeze()
            mse_val  = torch.mean((pred_val - y_val.to(DEVICE)) ** 2).item()

            oc_val = 0.0
            if qp_active and oc_scale is not None:
                pred_val_gw = prediction_to_gw(pred_val)
                Bv = pred_val_gw.shape[0]
                try:
                    h_bv = pred_val_gw.double().cpu().reshape(Bv, 1)
                    z_bv, s_bv = qp_layer(h_bv,
                                          cap_val_hourly,
                                          cost_cpu.unsqueeze(0).expand(Bv, -1))
                    served    = z_bv.sum(dim=1)
                    y_gw_bv   = ((y_val.to(DEVICE) * sig_y + mu_y) / 1000.0).double().cpu()
                    shortfall = torch.relu(y_gw_bv - served)
                    fuel      = (z_bv * cost_cpu.unsqueeze(0)).sum(dim=1)
                    smooth    = 0.1 * (z_bv ** 2).sum(dim=1)
                    oc_val    = (fuel + VOLL_OC * shortfall + smooth).mean().item()
                except Exception:
                    oc_val = 0.0

        mean_mse_train = np.mean(mse_batch)
        mean_oc_train  = np.mean(oc_batch)

        if qp_active and oc_scale is not None:
            train_loss = ALPHA * mean_mse_train + (1 - ALPHA) * mean_oc_train / oc_scale
            val_loss   = ALPHA * mse_val + (1 - ALPHA) * oc_val / oc_scale
        else:
            train_loss = mean_mse_train
            val_loss   = mse_val

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mse_train"].append(mean_mse_train)
        history["oc_train"].append(mean_oc_train)
        history["mse_val"].append(mse_val)
        history["oc_val"].append(oc_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1
            marker = ""
            if wait >= PATIENCE:
                print(f"  >>> Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

        ep_time = time.time() - ep_start
        status = "QP ON" if qp_active else "warmup"
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {mean_mse_train:>12.6f}  "
              f"{mean_oc_train:>12.4f}  {mse_val:>12.6f}  {status:>8}  {ep_time:>5.0f}s{marker}")

    training_time = time.time() - t_start
    total_epochs = len(history["epoch"])

    print(f"  Training: {total_epochs} epochs in {training_time / 60:.1f} min")
    if qp_fail_count > 0:
        print(f"  QP solver failures: {qp_fail_count}")

    # ── Load best weights ──
    model.load_state_dict(best_weights)
    model.eval()

    # ── Test predictions ──
    with torch.no_grad():
        pred_test_std = model(X_test.to(DEVICE)).squeeze()
        test_mse = torch.mean((pred_test_std - y_test_tensor.to(DEVICE)) ** 2).item()

        pred_ico = np.maximum(
            pred_test_std.cpu().numpy() * sig_y + mu_y, 1.0
        )

        # Test OC (batched, on CPU)
        pred_test_gw = prediction_to_gw(pred_test_std)
        Bt = pred_test_gw.shape[0]
        try:
            h_bt = pred_test_gw.double().cpu().reshape(Bt, 1)
            z_bt, s_bt = qp_layer(h_bt,
                                  cap_test_hourly,
                                  cost_cpu.unsqueeze(0).expand(Bt, -1))
            served    = z_bt.sum(dim=1)
            y_gw_bt   = torch.tensor(y_test, dtype=torch.float64) / 1000.0
            shortfall = torch.relu(y_gw_bt - served)
            fuel      = (z_bt * cost_cpu.unsqueeze(0)).sum(dim=1)
            smooth    = 0.1 * (z_bt ** 2).sum(dim=1)
            test_oc   = (fuel + VOLL_OC * shortfall + smooth).mean().item()
        except Exception:
            test_oc = 0.0
        test_loss = ALPHA * test_mse + (1 - ALPHA) * test_oc / oc_scale if oc_scale else test_mse

    print(f"  Test loss: {test_loss:.6f}  (MSE={test_mse:.6f}, OC={test_oc:.4f})")

    # ── QP Dispatch on test set ──
    z_dispatch = cp.Variable(n_sources)
    s_dispatch = cp.Variable(1)
    h_dispatch = cp.Parameter(1)
    cap_dispatch = cp.Parameter(n_sources)
    c_dispatch = cp.Parameter(n_sources)

    dispatch_obj = cp.Minimize(
        c_dispatch @ z_dispatch + 1000 * cp.sum(s_dispatch) + 0.1 * cp.sum_squares(z_dispatch)
    )
    dispatch_con = [
        z_dispatch >= 0,
        z_dispatch <= cap_dispatch,
        s_dispatch >= 0,
        cp.sum(z_dispatch) + s_dispatch == h_dispatch,
    ]

    N_test = len(y_test)
    curtailment = []
    qp_objectives = []
    dispatch_matrix = np.zeros((N_test, n_sources))

    for i in range(N_test):
        h_dispatch.value = np.array([max(float(pred_ico[i]), 0.1) / 1000])
        cap_dispatch.value = np.maximum(cap_test[i].astype(np.float64), 0) / 1000
        c_dispatch.value = cost_vec
        prob = cp.Problem(dispatch_obj, dispatch_con)
        prob.solve(solver=cp.ECOS)
        if prob.status in ["optimal", "optimal_inaccurate"] and z_dispatch.value is not None:
            z_opt = z_dispatch.value * 1000
            s_opt = max(s_dispatch.value[0] * 1000, 0.0)
            obj_val = float(prob.value)
        else:
            z_opt = np.zeros(n_sources)
            s_opt = float(pred_ico[i])
            obj_val = 9999.0

        curtailment.append(max(0.0, float(y_test[i]) - z_opt.sum()))
        qp_objectives.append(obj_val)
        dispatch_matrix[i] = z_opt

    curtailment = np.array(curtailment)
    qp_objectives = np.array(qp_objectives)

    # ── Compute all metrics ──
    err = pred_ico - y_test
    results = {
        "seed": seed,
        "r2": r2_score(y_test, pred_ico),
        "mae": mean_absolute_error(y_test, pred_ico),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_ico)),
        "mape": np.mean(np.abs(err) / y_test) * 100,
        "bias": float(err.mean()),
        "std_err": float(err.std()),
        "avg_curtailment": curtailment.mean(),
        "max_curtailment": curtailment.max(),
        "curtail_rate": (curtailment > 1.0).mean() * 100,
        "total_unserved": curtailment.sum(),
        "avg_qp_obj": qp_objectives.mean(),
        "total_dispatch": dispatch_matrix.sum(axis=1).mean(),
        "fossil_dispatch": dispatch_matrix[:, [i for i, g in enumerate(GEN_LABELS) if g in FOSSIL_LABELS]].sum(axis=1).mean(),
        "clean_dispatch": dispatch_matrix[:, [i for i, g in enumerate(GEN_LABELS) if g not in FOSSIL_LABELS]].sum(axis=1).mean(),
        "per_source_dispatch": dispatch_matrix.mean(axis=0),
        "epochs": total_epochs,
        "training_time": training_time,
        "oc_scale": oc_scale,
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_oc": test_oc,
        "pred_ico": pred_ico,
        "curtailment": curtailment,
        "qp_objectives": qp_objectives,
        "dispatch_matrix": dispatch_matrix,
        "history": history,
        "gradient_norms": gradient_norms,
        "best_weights": best_weights,
    }

    return results


# ── Run the single seed for this job ─────────────────────────
print("=" * 70)
print(f"  TRAINING ICO  |  SEED {SEED}")
print("=" * 70)
r = train_ico_single_seed(SEED)
print(f"  R2={r['r2']:.4f}  MAE={r['mae']:.0f} MW  Bias={r['bias']:+.1f} MW  "
      f"Curtail={r['avg_curtailment']:.1f} MW/hr")

# ── Save per-seed outputs ────────────────────────────────────
np.save(os.path.join(SAVE_DIR, f"pred_ico_seed{SEED}.npy"), r["pred_ico"])
np.save(os.path.join(SAVE_DIR, f"curtailment_seed{SEED}.npy"), r["curtailment"])
np.save(os.path.join(SAVE_DIR, f"dispatch_matrix_seed{SEED}.npy"), r["dispatch_matrix"])
np.save(os.path.join(SAVE_DIR, f"objectives_seed{SEED}.npy"), r["qp_objectives"])
torch.save(r["best_weights"], os.path.join(SAVE_DIR, f"ico_weights_seed{SEED}.pt"))
np.save(os.path.join(SAVE_DIR, f"history_seed{SEED}.npy"), r["history"])
np.save(os.path.join(SAVE_DIR, f"gradient_norms_seed{SEED}.npy"), r["gradient_norms"])
if not os.path.exists(os.path.join(SAVE_DIR, "y_test.npy")):
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

# ── Per-seed metrics CSV (one row; aggregate across seeds afterwards) ──
metrics_row = {k: r[k] for k in [
    "seed", "r2", "mae", "rmse", "mape", "bias", "std_err",
    "avg_curtailment", "max_curtailment", "curtail_rate", "total_unserved",
    "avg_qp_obj", "total_dispatch", "fossil_dispatch", "clean_dispatch",
    "epochs", "training_time", "oc_scale", "test_loss", "test_mse", "test_oc",
]}
metrics_row["training_time_min"] = r["training_time"] / 60.0
csv_path = os.path.join(SAVE_DIR, f"metrics_seed{SEED}.csv")
pd.DataFrame([metrics_row]).to_csv(csv_path, index=False)
print(f"\nSaved metrics -> {csv_path}")
print(f"Saved arrays and weights -> {SAVE_DIR}/  (seed {SEED})")
