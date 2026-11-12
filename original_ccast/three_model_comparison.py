#!/usr/bin/env python
# =====================================================================
# THREE-MODEL COMPARISON (ICO vs PtO vs LS) — single seed, real data.
# Controlled comparison: ICO and PtO share the SAME network, data,
# batch size and hyperparameters and differ ONLY in the training
# objective (composite vs MSE-only). LS is OLS on the same features.
# All three forecasts are scored through ONE identical dispatch.
# Reads SEED, DATA_DIR, SAVE_DIR from the env. Writes
# comparison_seed<SEED>.csv + per-model predictions to SAVE_DIR.
#   qsub -v SEED=42 run_compare.pbs
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
SAVE_DIR = os.environ.get("SAVE_DIR", "results/comparison")
SEED     = int(os.environ.get("SEED", "42"))
ENERGY_PATH  = os.path.join(DATA_DIR, "data", "modified_dataset.csv")
WEATHER_PATH = os.path.join(DATA_DIR, "data", "weather_features.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────
ALPHA      = 0.80        # composite loss: 80% MSE + 20% OC(QP)
MAX_EPOCHS = 100
BATCH_SIZE = 64          # every sample in batch goes through QP
PATIENCE   = 25
WARMUP_EP  = 5           # MSE-only warmup before QP activates
LR         = 5e-5
WEIGHT_DECAY = 2e-2

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

# ── Train / Validation / Test split (80/10/10 chronological) ─
n = len(df)
tr_end = int(n * 0.8)
vl_end = int(n * 0.9)

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
X_test  = torch.FloatTensor(X_scaled[vl_end:])
y_test_tensor = torch.FloatTensor(y_norm[vl_end:])   # standardized, for test loss
y_test  = y[vl_end:]                                   # original scale, for metrics

# ── Generator capacities and cost vector ─────────────────────
n_sources = len(GEN_COLS)
installed_cap = df[GEN_COLS].iloc[:tr_end].max().values.astype(np.float32)

capacities = np.zeros((n, n_sources), dtype=np.float32)
for j, col in enumerate(GEN_COLS):
    if col in VARIABLE_RENEWABLE:
        capacities[:, j] = np.maximum(
            df[col].rolling(24, min_periods=1).max().values * 1.2,
            0.05 * installed_cap[j]
        ).astype(np.float32)
    else:
        capacities[:, j] = installed_cap[j]

cap_test = capacities[vl_end:][-len(y_test):]

# Per-hour capacity slices (GW) for the embedded QP, so PtO TRAINS on the true
# hour-by-hour hybrid ceilings (Eq. 6), matching the fixed ICO model. Rows align
# to X_train / X_val / y_test. LS (OLS + merit) and the final evaluation loop
# already use hourly caps, so only PtO's training path needed this.
cap_train_hourly = torch.DoubleTensor(capacities[:tr_end].astype(np.float64) / 1000.0)
cap_val_hourly   = torch.DoubleTensor(capacities[tr_end:vl_end].astype(np.float64) / 1000.0)

cost_vec = np.array(
    [10.0 if c in FOSSIL_COLS else 0.1 for c in GEN_COLS],
    dtype=np.float32,
)

n_batches = int(np.ceil(tr_end / BATCH_SIZE))
print(f"Data: {n:,} hrs | Train {tr_end:,} | Val {vl_end - tr_end:,} | Test {len(y_test):,}")
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

def train_mlp(seed, alpha=ALPHA, use_oc=True, tag="ICO"):

    print(f"\n{'='*70}\n  TRAINING {tag}  |  seed {seed}  (alpha={alpha}, use_oc={use_oc})\n{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DemandForecaster(len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # Reproducible DataLoader shuffle. idx_train carries each sample's row index
    # so the QP can use that hour's capacity ceiling (cap_train_hourly[idx]).
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
        qp_active = use_oc and (epoch > WARMUP_EP)

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
                    total_loss = alpha * mse + (1 - alpha) * oc.float().to(DEVICE) / oc_scale
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
            train_loss = alpha * mean_mse_train + (1 - alpha) * mean_oc_train / oc_scale
            val_loss   = alpha * mse_val + (1 - alpha) * oc_val / oc_scale
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

    return model, history, oc_scale, best_weights


# ============================================================
# Shared dispatch + metrics — identical for every model
# ============================================================
def evaluate(pred_mw, seed, tag):
    """Score a test forecast (MW) through the same QP dispatch and metrics."""
    pred_mw = np.maximum(np.asarray(pred_mw, dtype=float), 1.0)

    z_d = cp.Variable(n_sources)
    s_d = cp.Variable(1)
    h_d = cp.Parameter(1)
    cap_d = cp.Parameter(n_sources)
    c_d = cp.Parameter(n_sources)
    obj = cp.Minimize(c_d @ z_d + 1000 * cp.sum(s_d) + 0.1 * cp.sum_squares(z_d))
    con = [z_d >= 0, z_d <= cap_d, s_d >= 0, cp.sum(z_d) + s_d == h_d]

    N = len(y_test)
    curt = np.zeros(N); objs = np.zeros(N); oc_hr = np.zeros(N)
    D = np.zeros((N, n_sources))
    for i in range(N):
        h_d.value = np.array([max(float(pred_mw[i]), 0.1) / 1000])
        cap_d.value = np.maximum(cap_test[i].astype(np.float64), 0) / 1000
        c_d.value = cost_vec
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.ECOS)
        if prob.status in ["optimal", "optimal_inaccurate"] and z_d.value is not None:
            z_opt = z_d.value * 1000; objs[i] = float(prob.value)
        else:
            z_opt = np.zeros(n_sources); objs[i] = 9999.0
        sh = max(0.0, float(y_test[i]) - z_opt.sum())
        zg = z_opt / 1000.0
        curt[i] = sh
        oc_hr[i] = float(cost_vec @ zg) + VOLL_OC * (sh / 1000.0) + 0.1 * float((zg ** 2).sum())
        D[i] = z_opt

    err = pred_mw - y_test
    f_idx = [k for k, g in enumerate(GEN_LABELS) if g in FOSSIL_LABELS]
    c_idx = [k for k, g in enumerate(GEN_LABELS) if g not in FOSSIL_LABELS]
    m = {
        "model": tag, "seed": seed,
        "r2": r2_score(y_test, pred_mw),
        "mae": mean_absolute_error(y_test, pred_mw),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_mw)),
        "mape": float(np.mean(np.abs(err) / y_test) * 100),
        "bias": float(err.mean()), "std_err": float(err.std()),
        "avg_curtailment": float(curt.mean()), "max_curtailment": float(curt.max()),
        "curtail_rate": float((curt > 1.0).mean() * 100), "total_unserved": float(curt.sum()),
        "avg_operational_cost": float(oc_hr.mean()), "avg_qp_obj": float(objs.mean()),
        "total_dispatch": float(D.sum(1).mean()),
        "fossil_dispatch": float(D[:, f_idx].sum(1).mean()),
        "clean_dispatch": float(D[:, c_idx].sum(1).mean()),
    }
    return m, D, pred_mw


def fit_ls(seed):
    """Ordinary least squares on the same scaled features (closed form)."""
    Xtr = np.column_stack([np.ones(tr_end), X_scaled[:tr_end]])
    beta, *_ = np.linalg.lstsq(Xtr, y[:tr_end], rcond=None)
    Xte = np.column_stack([np.ones(len(X_scaled) - vl_end), X_scaled[vl_end:]])
    return np.maximum(Xte @ beta, 1.0)


def test_forecast_mw(model):
    model.eval()
    with torch.no_grad():
        ps = model(X_test.to(DEVICE)).squeeze().cpu().numpy()
    return np.maximum(ps * sig_y + mu_y, 1.0)


# ============================================================
# Run the controlled comparison for this seed
# ============================================================
# Reuse the ICO forecasts already produced by ico_train_single_seed.py.
# Same network/data/seed -> same model, so the ICO column matches your
# standalone ICO results and we skip ~110 min of retraining per seed.
# Set REUSE_ICO=0 to retrain ICO inside this script instead.
ICO_DIR   = os.environ.get("ICO_DIR", os.path.join(os.path.dirname(SAVE_DIR.rstrip("/")), "ico_model"))
REUSE_ICO = os.environ.get("REUSE_ICO", "1") == "1"
ico_pred_path = os.path.join(ICO_DIR, f"pred_ico_seed{SEED}.npy")

print("=" * 70)
print(f"  THREE-MODEL COMPARISON  |  SEED {SEED}")
print("=" * 70)

ico_hist = None
if REUSE_ICO and os.path.exists(ico_pred_path):
    pred_ico = np.load(ico_pred_path)
    print(f"[ICO] reusing saved forecasts: {ico_pred_path}  (n={len(pred_ico)})")
    assert len(pred_ico) == len(y_test), (
        f"saved ICO length {len(pred_ico)} != test set {len(y_test)} -- data changed; "
        "rerun with REUSE_ICO=0")
else:
    if REUSE_ICO:
        print(f"[ICO] no saved forecasts at {ico_pred_path} -- training ICO here")
    ico_model, ico_hist, _, _ = train_mlp(SEED, alpha=ALPHA, use_oc=True, tag="ICO")
    pred_ico = test_forecast_mw(ico_model)

pto_model, pto_hist, _, _ = train_mlp(SEED, alpha=1.0, use_oc=False, tag="PtO")

forecasts = {
    "ICO": pred_ico,
    "PtO": test_forecast_mw(pto_model),
    "LS":  fit_ls(SEED),
}

rows = []
for name, pred in forecasts.items():
    m, D, p = evaluate(pred, SEED, name)
    rows.append(m)
    np.save(os.path.join(SAVE_DIR, f"pred_{name}_seed{SEED}.npy"), p)
    np.save(os.path.join(SAVE_DIR, f"dispatch_{name}_seed{SEED}.npy"), D)

np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
np.save(os.path.join(SAVE_DIR, f"history_PtO_seed{SEED}.npy"), pto_hist, allow_pickle=True)
if ico_hist is not None:
    np.save(os.path.join(SAVE_DIR, f"history_ICO_seed{SEED}.npy"), ico_hist, allow_pickle=True)

cols = ["model", "seed", "r2", "mae", "rmse", "mape", "bias", "std_err",
        "avg_curtailment", "max_curtailment", "curtail_rate", "total_unserved",
        "avg_operational_cost", "avg_qp_obj", "total_dispatch",
        "fossil_dispatch", "clean_dispatch"]
df = pd.DataFrame(rows)[cols]
csv_path = os.path.join(SAVE_DIR, f"comparison_seed{SEED}.csv")
df.to_csv(csv_path, index=False)

print("\n" + "=" * 70)
print(f"  RESULTS  |  seed {SEED}")
print("=" * 70)
print(df[["model", "r2", "mae", "bias", "avg_curtailment",
          "avg_operational_cost"]].to_string(index=False))
print(f"\nSaved -> {csv_path}")
