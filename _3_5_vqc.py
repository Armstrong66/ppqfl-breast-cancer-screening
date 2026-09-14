"""
=============================================================================
3–5 — VARIATIONAL QUANTUM CIRCUIT (VQC) TRAINING & ABLATION
=============================================================================
Project : Quantum-Enhanced Hybrid Architectures for Mammographic Breast
          Cancer Classification in African and MENA Populations
Coverage:
  3 — VQC design + Regime A (frozen classical, train VQC only)
  4 — Regime B (end-to-end joint training); noise robustness evaluation
  5 — Hyperparameter sweep; ablation table vs classical baseline

Pipeline:
  PCA-compressed features (from 2b_feature_pca.py)
    → MinMaxScaler [0,1]
    → Angle encoding into n_qubits
    → Hardware-efficient ansatz (RY + circular CNOT)
    → Expectation value ⟨Z₀⟩ → sigmoid → binary classification

Outputs (../ppqfl-breast-cancer-screening/outputs/vqc_outputs/):
  regime_A/           ← frozen classical + train VQC
  regime_B/           ← end-to-end (fine-tune CNN head + VQC jointly)
  sweep/              ← hyperparameter sweep results
  noise/              ← robustness under Gaussian noise
  ablation_table.csv  ← full comparison table for reporting
  ablation_table.png  ← visual summary

Environment: CPU (quantum simulation — no GPU needed for VQC)
             PennyLane + PyTorch — install: pip install pennylane
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os, json, pickle, shutil, warnings, itertools
from pathlib import Path
from copy import deepcopy
from typing import Union, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pennylane as qml
from pennylane import numpy as pnp

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report, roc_curve, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss
)

from pipeline_utils import seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature inputs (from _2b_feature_pca.py) ───────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
BASE          = PROJECT_ROOT / "outputs"
FEAT_DIR      = BASE / "feature_outputs"
OUT_DIR       = BASE / "vqc_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for d in ["regime_A", "regime_B", "noise", "sweep", "reupload_ablation", "cv_outputs"]:
    (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

BASELINE_JSON = BASE / "baseline_outputs/baseline_results.json"

# ── VQC hyperparameter grid (5 sweep) ───────────────────────────────────
# To run quickly: keep this small. Full sweep can be large.
SWEEP_CFG = {
    "n_qubits":    [4, 6, 8],       # maps to PCA n_components
    "n_layers":    [1, 2, 3],       # VQC depth (ansatz repetitions)
    "encoding":    ["angle"],       # "angle" only for now; extend later
    "lr":          [0.01, 0.005],
    "reupload":    [True, False],   # compare reupload vs no-reupload across the sweep
}

# ── Training config (shared across regimes) ──────────────────────────────────
TRAIN_CFG = {
    "batch_size":    32,
    "num_epochs":    50,            # VQC converges slower; more epochs needed
    "patience":      10,
    "random_state":  42,
}

# ── Noise robustness evaluation ───────────────────────────────────────────────
# Gaussian noise levels added to raw images before feature extraction
# (applied directly to the PCA features as a proxy here)
NOISE_SIGMAS = [0.0, 0.05, 0.10, 0.20]

DEVICE = torch.device("cpu")   # VQC simulation runs on CPU


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_split(n_qubits: int, noise_sigma: float = 0.0):
    """
    Load PCA-compressed features for the requested qubit count,
    apply MinMaxScaler [0,1] (fitted on train only — no leakage),
    optionally inject Gaussian noise (for robustness evaluation),
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler).
    """
    n = n_qubits  # PCA n_components == n_qubits for angle encoding

    X_train = np.load(FEAT_DIR / f"features_train_pca{n}.npy")
    X_val   = np.load(FEAT_DIR / f"features_val_pca{n}.npy")
    X_test  = np.load(FEAT_DIR / f"features_test_pca{n}.npy")
    y_train = np.load(FEAT_DIR / "labels_train.npy")
    y_val   = np.load(FEAT_DIR / "labels_val.npy")
    y_test  = np.load(FEAT_DIR / "labels_test.npy")

    # MinMaxScaler fitted on train only
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Optional noise injection (robustness experiment)
    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        X_val  = np.clip(X_val  + rng.normal(0, noise_sigma, X_val.shape),  0, 1)
        X_test = np.clip(X_test + rng.normal(0, noise_sigma, X_test.shape), 0, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    to_t = lambda x, y: TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    train_dl = DataLoader(to_t(X_train, y_train), batch_size=batch_size,
                          shuffle=True,  drop_last=False)
    val_dl   = DataLoader(to_t(X_val,   y_val),   batch_size=batch_size,
                          shuffle=False, drop_last=False)
    test_dl  = DataLoader(to_t(X_test,  y_test),  batch_size=batch_size,
                          shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VQC ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_vqc(n_qubits: int, n_layers: int, reupload: bool = True):
    """
    Build a PennyLane VQC as a PyTorch-compatible nn.Module layer.

    Circuit structure (repeated n_layers times):
      1. Angle encoding:   RY(π·x_i) on qubit i (if layer==0 or reupload==True)
      2. Variational layer: RY(θ_i) on each qubit
      3. Entanglement:      CNOT in circular pattern (0→1→2→...→n-1→0)

    Measurement: expectation value ⟨Z₀⟩ on qubit 0 → scalar in [-1, 1]
    Mapped to [0, 1] probability via sigmoid(⟨Z₀⟩ + bias) for BCE loss.

    Data re-uploading (reupload=True) allows the VQC to express higher-frequency
    Fourier components, increasing non-linear expressivity.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            if layer == 0 or reupload:
                for i in range(n_qubits):
                    qml.RY(np.pi * inputs[i], wires=i)
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HQCNNClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network:
      Input (n_qubits,) → VQC → scalar ⟨Z₀⟩ → sigmoid → P(malignant)
    """
    def __init__(self, n_qubits: int, n_layers: int, reupload: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.vqc      = build_vqc(n_qubits, n_layers, reupload=reupload)
        self.bias     = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        return torch.sigmoid(out + self.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassicalMicroMLP(nn.Module):
    """Minimal classical MLP baseline matched to the VQC parameter budget."""
    def __init__(self, input_dim: int, hidden_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def find_optimal_threshold_youden(y_true, probs) -> float:
    """Find optimal classification threshold maximizing Youden's J = TPR - FPR on validation set."""
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    return float(np.clip(best_thresh, 0.05, 0.95))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        probs  = model(X_batch)
        loss   = criterion(probs, y_batch.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        preds_all.extend((probs.detach() > 0.5).long().numpy())
        labels_all.extend(y_batch.numpy())
    n    = len(loader.dataset)
    acc  = accuracy_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all, zero_division=0)
    return total_loss / n, acc, f1


@torch.no_grad()
def eval_epoch(model, loader, criterion, opt_threshold: float = None):
    model.eval()
    total_loss, probs_all, labels_all = 0.0, [], []
    for X_batch, y_batch in loader:
        probs  = model(X_batch)
        loss   = criterion(probs, y_batch.float())
        total_loss += loss.item() * X_batch.size(0)
        probs_all.extend(probs.numpy())
        labels_all.extend(y_batch.numpy())
    n   = len(loader.dataset)
    probs_np = np.array(probs_all)
    labels_np = np.array(labels_all)
    
    # Threshold metrics: standard 0.5
    preds_05 = (probs_np > 0.5).astype(int)
    acc_05 = accuracy_score(labels_np, preds_05)
    f1_05  = f1_score(labels_np, preds_05, zero_division=0)
    
    # Threshold metrics: optimal threshold
    tau = opt_threshold if opt_threshold is not None else find_optimal_threshold_youden(labels_np, probs_np)
    preds_opt = (probs_np > tau).astype(int)
    acc_opt = accuracy_score(labels_np, preds_opt)
    f1_opt  = f1_score(labels_np, preds_opt, zero_division=0)
    bal_acc = balanced_accuracy_score(labels_np, preds_opt) if len(set(labels_np)) > 1 else acc_opt
    mcc_opt = matthews_corrcoef(labels_np, preds_opt) if len(set(labels_np)) > 1 else 0.0

    auc = roc_auc_score(labels_np, probs_np) if len(set(labels_np)) > 1 else 0.0
    aupr = average_precision_score(labels_np, probs_np) if len(set(labels_np)) > 1 else 0.0
    return total_loss / n, acc_05, f1_05, auc, aupr, labels_all, preds_opt, probs_all, f1_opt, tau, bal_acc, mcc_opt


def train_vqc(n_qubits: int, n_layers: int, lr: float,
              train_dl, val_dl, out_subdir: Path,
              label: str = "", noise_sigma: float = 0.0,
              reupload: bool = True) -> dict:
    """
    Full training run for one VQC configuration.
    Returns a results dict with test metrics and parameter count.
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    model     = HQCNNClassifier(n_qubits, n_layers, reupload=reupload)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-4
    )

    best_auc, best_aupr, best_state, patience_ctr = 0.0, 0.0, None, 0
    history = []

    print(f"    Training: qubits={n_qubits} layers={n_layers} lr={lr} reupload={reupload} {label}")
    for epoch in range(1, TRAIN_CFG["num_epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, vl_auc, vl_aupr, _, _, _, vl_f1_opt, _, _, _ = eval_epoch(model, val_dl, criterion)
        scheduler.step(vl_auc)

        is_best = vl_auc > best_auc
        if is_best:
            best_auc   = vl_auc
            best_aupr  = vl_aupr
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_auc": vl_auc,
            "val_aupr": vl_aupr, "val_f1": vl_f1, "val_f1_opt": vl_f1_opt,
            "is_best": is_best,
        })
        if epoch % 10 == 0 or is_best:
            print(f"      Ep {epoch:3d} | TrLoss {tr_loss:.4f} | "
                  f"VlAUC {vl_auc:.4f} | VlF1(τ*) {vl_f1_opt:.4f}" + (" ← best" if is_best else ""))

        if patience_ctr >= TRAIN_CFG["patience"]:
            print(f"      Early stopping at epoch {epoch}.")
            break

    # Save checkpoint + history
    suffix = "" if reupload else "_noreupload"
    ckpt_name = f"vqc_q{n_qubits}_l{n_layers}_lr{lr}{suffix}.pt"
    torch.save(best_state, out_subdir / ckpt_name)
    pd.DataFrame(history).to_csv(out_subdir / ckpt_name.replace(".pt", "_history.csv"),
                                 index=False, encoding="utf-8-sig")

    return {
        "n_qubits":      n_qubits,
        "n_layers":      n_layers,
        "lr":            lr,
        "reupload":      reupload,
        "noise_sigma":   noise_sigma,
        "best_val_auc":  round(best_auc, 4),
        "best_val_aupr": round(best_aupr, 4),
        "vqc_params":    model.count_params(),
        "best_state":    best_state,    # kept in memory for test eval
        "model_ref":     model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION & PLOTTING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluate(result: dict, test_dl, val_dl=None) -> dict:
    """Load best weights, calibrate threshold on validation set, and evaluate on test set."""
    model = result["model_ref"]
    model.load_state_dict(result["best_state"])
    criterion = nn.BCELoss()
    
    # Tune optimal threshold on validation set if provided
    opt_tau = 0.5
    if val_dl is not None:
        _, _, _, _, _, _, _, val_probs, _, opt_tau, _, _ = eval_epoch(model, val_dl, criterion)

    _, ts_acc, ts_f1, ts_auc, ts_aupr, y_true, y_pred, y_prob, ts_f1_opt, _, ts_bal_acc, ts_mcc = eval_epoch(
        model, test_dl, criterion, opt_threshold=opt_tau
    )
    return {
        **{k: v for k, v in result.items() if k not in ("best_state", "model_ref")},
        "test_accuracy":     round(ts_acc, 4),
        "test_f1":           round(ts_f1,  4),
        "test_f1_opt":       round(ts_f1_opt, 4),
        "opt_threshold":     round(opt_tau, 4),
        "test_balanced_acc": round(ts_bal_acc, 4),
        "test_mcc":          round(ts_mcc, 4),
        "test_auc_roc":      round(ts_auc, 4),
        "test_aupr":         round(ts_aupr, 4),
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


def run_classical_control_configs(n_features: int = 4) -> list:
    """Train and evaluate classical control models for the ablation table."""
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_features)
    train_dl, val_dl, test_dl = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        TRAIN_CFG["batch_size"]
    )

    results = []

    # Logistic regression baseline (simple linear classifier)
    logreg = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_probs = logreg.predict_proba(X_test)[:, 1]
    logreg_preds = (logreg_probs > 0.5).astype(int)
    val_probs = logreg.predict_proba(X_val)[:, 1]
    test_aupr = average_precision_score(y_test, logreg_probs) if len(set(y_test)) > 1 else 0.0
    val_aupr = average_precision_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
    results.append({
        "model": "Logistic Regression",
        "regime": "Classical control",
        "n_qubits": n_features,
        "n_layers": 0,
        "trainable_params": int(logreg.coef_.size + logreg.intercept_.size),
        "noise_sigma": 0.0,
        "best_val_auc": round(roc_auc_score(y_val, val_probs), 4),
        "best_val_aupr": round(val_aupr, 4),
        "test_auc_roc": round(roc_auc_score(y_test, logreg_probs), 4),
        "test_aupr": round(test_aupr, 4),
        "test_f1": round(f1_score(y_test, logreg_preds, zero_division=0), 4),
        "test_accuracy": round(accuracy_score(y_test, logreg_preds), 4),
        "notes": "Classical logistic regression control",
    })

    # Micro-MLP baseline matched to a small parameter budget
    model = ClassicalMicroMLP(n_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_val_auc, patience_ctr, best_state = 0.0, 0, None

    for epoch in range(1, TRAIN_CFG["num_epochs"] + 1):
        model.train()
        for X_b, y_b in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b.float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= TRAIN_CFG["patience"]:
            break

    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    model.eval()
    with torch.no_grad():
        test_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    test_preds = (test_probs > 0.5).astype(int)
    test_aupr = average_precision_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.0
    val_aupr = average_precision_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0

    torch.save(model.state_dict(), OUT_DIR / "micromlp_q4.pt")
    results.append({
        "model": "Micro-MLP",
        "regime": "Classical control",
        "n_qubits": n_features,
        "n_layers": 0,
        "trainable_params": model.count_params(),
        "noise_sigma": 0.0,
        "best_val_auc": round(best_val_auc, 4),
        "best_val_aupr": round(val_aupr, 4),
        "test_auc_roc": round(roc_auc_score(y_test, test_probs), 4),
        "test_aupr": round(test_aupr, 4),
        "test_f1": round(f1_score(y_test, test_preds, zero_division=0), 4),
        "test_accuracy": round(accuracy_score(y_test, test_preds), 4),
        "notes": "Classical micro-MLP control (≈9–11 params)",
    })
    return results


def _select_hidden_dim_for_budget(target_params: int, input_dim: int = 4) -> int:
    """Choose a compact classical MLP hidden size that approximates the VQC budget."""
    if target_params <= 0:
        return 1
    hidden = max(1, round((target_params - 1) / (input_dim + 2)))
    candidates = sorted({max(1, hidden - 1), hidden, hidden + 1})
    best_hidden = min(candidates, key=lambda h: abs(((input_dim + 2) * h + 1) - target_params))
    return best_hidden


class ClassicalParamMatchedMLP(nn.Module):
    """Classical MLP baseline matched to a target parameter budget."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_parameter_matched_mlp_baselines(top_runs: list) -> list:
    """Train classical MLP baselines matched to the top VQC sweep configuration budgets."""
    results = []
    for idx, run in enumerate(top_runs, start=1):
        n_qubits = run["n_qubits"]
        target_params = run["vqc_params"]
        hidden_dim = _select_hidden_dim_for_budget(target_params, input_dim=n_qubits)

        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_qubits)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )

        model = ClassicalParamMatchedMLP(n_qubits, hidden_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        best_val_auc, patience_ctr, best_state = 0.0, 0, None
        print(f"    Training classical MLP match {idx}: q={n_qubits} hidden={hidden_dim} "
              f"(target {target_params} params)")

        for epoch in range(1, TRAIN_CFG["num_epochs"] + 1):
            model.train()
            for X_b, y_b in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b.float())
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
            val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= TRAIN_CFG["patience"]:
                break

        model.load_state_dict(best_state if best_state is not None else model.state_dict())
        model.eval()
        with torch.no_grad():
            val_probs = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
            test_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            
        tau_mlp = find_optimal_threshold_youden(y_val, val_probs)
        test_preds = (test_probs > 0.5).astype(int)
        test_preds_opt = (test_probs > tau_mlp).astype(int)
        val_aupr = average_precision_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
        test_aupr = average_precision_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.0

        results.append({
            "model": f"Classical MLP match q={n_qubits} h={hidden_dim}",
            "regime": "Classical baseline (param-matched)",
            "n_qubits": n_qubits,
            "n_layers": 0,
            "trainable_params": model.count_params(),
            "noise_sigma": 0.0,
            "best_val_auc": round(best_val_auc, 4),
            "best_val_aupr": round(val_aupr, 4),
            "test_auc_roc": round(roc_auc_score(y_test, test_probs), 4),
            "test_aupr": round(test_aupr, 4),
            "test_f1": round(f1_score(y_test, test_preds, zero_division=0), 4),
            "test_f1_opt": round(f1_score(y_test, test_preds_opt, zero_division=0), 4),
            "opt_threshold": round(tau_mlp, 4),
            "test_accuracy": round(accuracy_score(y_test, test_preds), 4),
            "notes": f"Param-matched classical MLP for VQC q={n_qubits} l={run['n_layers']}",
        })
    return results


def plot_training_curve(history_csv: Path, title: str, save_path: Path):
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontweight="bold")
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="#4C72B0")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="#DD8452", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss"); axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(df["epoch"], df["val_auc"], color="#2ca02c")
    best_row = df.loc[df["val_auc"].idxmax()]
    axes[1].axvline(best_row["epoch"], color="red", linestyle=":", alpha=0.7,
                    label=f"Best: {best_row['val_auc']:.4f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val AUC-ROC"); axes[1].legend()
    axes[1].set_title("Validation AUC-ROC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_prob, label: str, save_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#4C72B0", label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"k--", alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {label}"); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ABLATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_ablation_table(all_results: list, baseline_json: Path) -> pd.DataFrame:
    """
    Assemble the full ablation table including the classical baseline row.
    Columns: Model, Qubits, Layers, TrainableParams, ValAUC, TestAUC, TestF1, TestAcc, Notes
    """
    rows = []

    # ── Classical baseline row ────────────────────────────────────────────────
    if baseline_json.exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        rows.append({
            "Model":           bl.get("backbone", bl.get("model", "Classical")),
            "Regime":          "Classical (head-only fine-tune)",
            "Qubits":          "N/A",
            "Layers":          "N/A",
            "TrainableParams": bl.get("trainable_params", bl.get("head_params", "N/A")),
            "NoiseSigma":      0.0,
            "ValAUC":          bl.get("best_val_auc", "N/A"),
            "ValAUPRC":       bl.get("best_val_aupr", "N/A"),
            "TestAUC":         bl.get("test_auc_roc", "N/A"),
            "TestAUCPR":      bl.get("test_aupr", "N/A"),
            "TestF1":          bl.get("test_f1",      "N/A"),
            "TestAcc":         bl.get("test_accuracy","N/A"),
            "Notes":           "Classical baseline (frozen backbone)",
        })

    # ── VQC & control result rows ──────────────────────────────────────────────
    for r in all_results:
        model_label = r.get("model") or f"HQCNN (VQC q={r.get('n_qubits', 'N/A')} l={r.get('n_layers', 'N/A')})"
        regime = str(r.get("regime", "A — frozen classical + VQC"))
        is_classical = "control" in regime.lower() or "classical" in model_label.lower() or "logistic" in model_label.lower()
        rows.append({
            "Model":           model_label,
            "Regime":          regime,
            "Qubits":          "N/A" if is_classical else r.get("n_qubits", "N/A"),
            "Layers":          "N/A" if is_classical else r.get("n_layers", "N/A"),
            "TrainableParams": r.get("trainable_params", r.get("vqc_params", "N/A")),
            "NoiseSigma":      r.get("noise_sigma", 0.0),
            "ValAUC":          r.get("best_val_auc", r.get("val_auc", "N/A")),
            "ValAUPRC":       r.get("best_val_aupr", "N/A"),
            "TestAUC":         r.get("test_auc_roc", r.get("test_auc", "N/A")),
            "TestAUCPR":      r.get("test_aupr", "N/A"),
            "TestF1":          r.get("test_f1",      "N/A"),
            "TestAcc":         r.get("test_accuracy","N/A"),
            "Notes":           r.get("notes", ""),
        })

    # ── Deduplicate duplicate rows (e.g. reupload ablation identical to sweep) ──
    df = pd.DataFrame(rows)
    # Deduplicate while preserving order
    dedup_cols = ["Model", "Regime", "Qubits", "Layers", "TrainableParams", "NoiseSigma", "ValAUC", "TestAUC"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    return df


def plot_ablation_table(df: pd.DataFrame, save_path: Path):
    """Render the ablation DataFrame as a publication-quality figure."""
    display_cols = ["Model", "Regime", "Qubits", "Layers",
                    "TrainableParams", "NoiseSigma",
                    "ValAUC", "ValAUPRC", "TestAUC", "TestAUCPR", "TestF1", "TestAcc"]
    plot_df = df[display_cols].copy()
    plot_df = plot_df.fillna("N/A")

    fig, ax = plt.subplots(figsize=(20, max(4.5, len(plot_df) * 0.65 + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.7)

    # Style header
    for j in range(len(display_cols)):
        tbl[(0, j)].set_facecolor("#1F4E79")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(plot_df) + 1):
        color = "#EBF3FB" if i % 2 == 0 else "white"
        for j in range(len(display_cols)):
            tbl[(i, j)].set_facecolor(color)

    ax.set_title("Ablation Table — HQCNN vs Classical Baseline\n"
                 "Mendeley Mammogram Dataset (Polokwane, South Africa)",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Ablation table saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REGIME A — Frozen classical + train VQC only (3)
# ══════════════════════════════════════════════════════════════════════════════

def run_regime_A() -> list:
    """
    Regime A: classical backbone frozen, only VQC parameters trained.
    """
    print("\n" + "═"*70)
    print("  REGIME A — Frozen Classical + Train VQC")
    print("═"*70)
    out = OUT_DIR / "regime_A"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    seed_everything(42)

    for n_qubits, n_layers, lr in [(4, 2, 0.01), (6, 2, 0.01), (8, 2, 0.01)]:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_qubits)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )
        res = train_vqc(n_qubits, n_layers, lr, train_dl, val_dl, out,
                        label="[Regime A]", reupload=True)
        res["regime"] = "A — frozen classical + VQC"
        res = test_evaluate(res, test_dl, val_dl=val_dl)

        ckpt_name = f"vqc_q{n_qubits}_l{n_layers}_lr{lr}"
        history_csv = out / f"{ckpt_name}_history.csv"
        if history_csv.exists():
            plot_training_curve(
                history_csv,
                f"Regime A — q={n_qubits} l={n_layers}",
                out / f"{ckpt_name}_curves.png"
            )
        plot_confusion_matrix(
            res["y_true"], res["y_pred"],
            f"Regime A — q={n_qubits} l={n_layers}",
            out / f"{ckpt_name}_cm.png"
        )
        plot_roc(
            res["y_true"], res["y_prob"],
            f"HQCNN Regime A q={n_qubits}",
            out / f"{ckpt_name}_roc.png"
        )
        print(f"    Test AUC={res['test_auc_roc']:.4f} F1(0.5)={res['test_f1']:.4f} F1(τ*)={res['test_f1_opt']:.4f} "
              f"Params={res['vqc_params']}")
        results.append(res)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REGIME B — End-to-end joint training (4)
# ══════════════════════════════════════════════════════════════════════════════

def run_regime_B() -> list:
    """
    Regime B: CNN head + VQC parameters optimised jointly.
    """
    print("\n" + "═"*70)
    print("  REGIME B — End-to-End Joint Training (CNN projection + VQC)")
    print("═"*70)
    out = OUT_DIR / "regime_B"
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for n_qubits, n_layers, lr in [(4, 2, 0.005), (6, 2, 0.005), (8, 2, 0.005)]:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_qubits)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )

        class RegimeBModel(nn.Module):
            def __init__(self, in_dim, n_qubits, n_layers):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(in_dim, n_qubits),
                    nn.Sigmoid(),
                )
                self.hqcnn = HQCNNClassifier(n_qubits, n_layers, reupload=True)

            def forward(self, x):
                return self.hqcnn(self.proj(x))

            def count_params(self):
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

        seed_everything(42)
        model     = RegimeBModel(n_qubits, n_qubits, n_layers)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_auc, best_aupr, best_state, patience_ctr = 0.0, 0.0, None, 0
        history = []
        print(f"    Training: qubits={n_qubits} layers={n_layers} lr={lr} [Regime B]")

        for epoch in range(1, TRAIN_CFG["num_epochs"] + 1):
            model.train()
            for X_batch, y_batch in train_dl:
                optimizer.zero_grad()
                probs = model(X_batch)
                loss  = criterion(probs, y_batch.float())
                loss.backward(); optimizer.step()

            model.eval()
            vl_probs, vl_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    vl_probs.extend(model(X_batch).numpy())
                    vl_labels.extend(y_batch.numpy())
            vl_auc = roc_auc_score(vl_labels, vl_probs) if len(set(vl_labels)) > 1 else 0.0
            vl_aupr = average_precision_score(vl_labels, vl_probs) if len(set(vl_labels)) > 1 else 0.0
            scheduler.step(vl_auc)
 
            is_best = vl_auc > best_auc
            if is_best:
                best_auc = vl_auc
                best_aupr = vl_aupr
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            history.append({"epoch": epoch, "val_auc": vl_auc, "is_best": is_best})
            if epoch % 10 == 0 or is_best:
                print(f"      Ep {epoch:3d} | VlAUC {vl_auc:.4f}"
                      + (" ← best" if is_best else ""))
            if patience_ctr >= TRAIN_CFG["patience"]:
                print(f"      Early stopping at epoch {epoch}.")
                break

        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(
            out / f"regimeB_q{n_qubits}_l{n_layers}_history.csv", index=False, encoding="utf-8-sig"
        )
        torch.save(best_state, out / f"vqc_q{n_qubits}_l{n_layers}_lr{lr}.pt")

        # Test evaluation with validation-tuned threshold
        model.load_state_dict(best_state)
        model.eval()
        
        # Determine optimal threshold on validation set
        vl_probs_all, vl_labels_all = [], []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                vl_probs_all.extend(model(X_b).numpy())
                vl_labels_all.extend(y_b.numpy())
        tau_b = find_optimal_threshold_youden(vl_labels_all, vl_probs_all)

        ts_probs, ts_preds, ts_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_dl:
                p = model(X_batch)
                ts_probs.extend(p.numpy())
                ts_preds.extend((p > 0.5).long().numpy())
                ts_labels.extend(y_batch.numpy())

        ts_probs_np = np.array(ts_probs)
        ts_labels_np = np.array(ts_labels)
        ts_preds_opt = (ts_probs_np > tau_b).astype(int)
        
        ts_auc = roc_auc_score(ts_labels_np, ts_probs_np) if len(set(ts_labels_np)) > 1 else 0.0
        ts_aupr = average_precision_score(ts_labels_np, ts_probs_np) if len(set(ts_labels_np)) > 1 else 0.0
        ts_f1  = f1_score(ts_labels_np, (ts_probs_np > 0.5).astype(int), zero_division=0)
        ts_f1_opt = f1_score(ts_labels_np, ts_preds_opt, zero_division=0)
        ts_acc = accuracy_score(ts_labels_np, ts_preds_opt)
 
        res = {
            "n_qubits": n_qubits, "n_layers": n_layers, "lr": lr,
            "noise_sigma": 0.0,
            "regime": "B — end-to-end (projection + VQC)",
            "best_val_auc": round(best_auc, 4),
            "best_val_aupr": round(best_aupr, 4),
            "vqc_params": model.count_params(),
            "test_auc_roc": round(ts_auc, 4),
            "test_aupr": round(ts_aupr, 4),
            "test_f1":      round(ts_f1,  4),
            "test_f1_opt":  round(ts_f1_opt, 4),
            "opt_threshold": round(tau_b, 4),
            "test_accuracy":round(ts_acc, 4),
            "y_true": ts_labels, "y_pred": ts_preds_opt, "y_prob": ts_probs,
        }
        plot_confusion_matrix(
            ts_labels, ts_preds_opt,
            f"Regime B — q={n_qubits} l={n_layers}",
            out / f"regimeB_q{n_qubits}_l{n_layers}_cm.png"
        )
        print(f"    Test AUC={ts_auc:.4f} F1(0.5)={ts_f1:.4f} F1(τ*)={ts_f1_opt:.4f} Params={model.count_params()}")
        results.append(res)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8.  HYPERPARAMETER SWEEP (5)
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep() -> list:
    """
    Grid sweep over SWEEP_CFG. Runs Regime A for each combination, including reuploading vs no-reuploading.
    """
    print("\n" + "═"*70)
    print("  HYPERPARAMETER SWEEP (Regime A: Qubits × Layers × LR × Re-uploading)")
    print("═"*70)
    out = OUT_DIR / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    results = []

    grid = list(itertools.product(
        SWEEP_CFG["n_qubits"],
        SWEEP_CFG["n_layers"],
        SWEEP_CFG["lr"],
        SWEEP_CFG.get("reupload", [True, False]),
    ))
    print(f"  Total configurations in sweep grid: {len(grid)}")

    for n_qubits, n_layers, lr, reup in grid:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_qubits)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )
        res = train_vqc(n_qubits, n_layers, lr, train_dl, val_dl, out,
                        label=f"[Sweep {'reupload' if reup else 'no-reupload'}]", reupload=reup)
        res["regime"] = f"A — sweep ({'reupload' if reup else 'no-reupload'})"
        res = test_evaluate(res, test_dl, val_dl=val_dl)
        res["notes"] = f"sweep lr={lr} reup={reup}"
        results.append(res)

    _plot_sweep_heatmap(results, out)

    best_result = max(results, key=lambda r: r["best_val_auc"])
    reup_suffix = "" if best_result.get("reupload", True) else "_noreupload"
    best_ckpt_name = f"vqc_q{best_result['n_qubits']}_l{best_result['n_layers']}_lr{best_result['lr']}{reup_suffix}.pt"
    best_ckpt_src  = out / best_ckpt_name
    regime_A_dir   = OUT_DIR / "regime_A"
    regime_A_dir.mkdir(parents=True, exist_ok=True)
    if best_ckpt_src.exists():
        shutil.copy2(best_ckpt_src, regime_A_dir / best_ckpt_name)
        # Also copy as canonical default checkpoint
        canonical_name = f"vqc_q{best_result['n_qubits']}_l{best_result['n_layers']}_lr{best_result['lr']}.pt"
        shutil.copy2(best_ckpt_src, regime_A_dir / canonical_name)
        print(f"  Promoted best sweep checkpoint to regime_A: {best_ckpt_name}")
    else:
        print(f"  [WARN] Best sweep checkpoint not found: {best_ckpt_src}")

    manifest = regime_A_dir / "best_run_manifest.json"
    with open(manifest, "w") as f:
        json.dump({
            "n_qubits": int(best_result["n_qubits"]),
            "n_layers": int(best_result["n_layers"]),
            "lr": float(best_result["lr"]),
            "reupload": bool(best_result.get("reupload", True)),
            "val_auc": float(best_result["best_val_auc"]),
            "regime": "A — sweep",
        }, f, indent=2)
    print(f"  Wrote best-run manifest: {manifest}")

    top_runs = select_top_sweep_configs(results, top_k=3)
    print("  Selected top-3 sweep configs by compact budget and validation performance:")
    for r in top_runs:
        print(f"    q={r['n_qubits']} l={r['n_layers']} reup={r.get('reupload', True)} params={r['vqc_params']} "
              f"val_auc={r['best_val_auc']:.4f} test_aupr={r.get('test_aupr', 0.0):.4f}")
    top_manifest = regime_A_dir / "top_sweep_manifest.json"
    with open(top_manifest, "w") as f:
        json.dump([
            {
                "n_qubits": int(r["n_qubits"]),
                "n_layers": int(r["n_layers"]),
                "lr": float(r["lr"]),
                "reupload": bool(r.get("reupload", True)),
                "val_auc": float(r["best_val_auc"]),
                "test_auc_roc": float(r["test_auc_roc"]),
                "test_aupr": float(r.get("test_aupr", 0.0)),
                "vqc_params": int(r["vqc_params"]),
                "regime": r["regime"],
                "notes": r.get("notes", ""),
            }
            for r in top_runs
        ], f, indent=2)
    print(f"  Wrote top-3 sweep manifest: {top_manifest}")
    return results


def _plot_sweep_heatmap(results: list, out: Path):
    records = [{"n_qubits": r["n_qubits"], "n_layers": r["n_layers"],
                "lr": r["lr"], "reupload": r.get("reupload", True), "val_auc": r["best_val_auc"]} for r in results]
    df = pd.DataFrame(records)
    out.mkdir(parents=True, exist_ok=True)
    
    if df["reupload"].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for idx, (reup_val, title) in enumerate([(True, "Data Re-uploading (Every Layer)"), (False, "No Re-uploading (Layer 0 Only)")]):
            sub_df = df[df["reupload"] == reup_val]
            if not sub_df.empty:
                best_lr = sub_df.groupby(["n_qubits","n_layers"])["val_auc"].max().reset_index()
                pivot   = best_lr.pivot(index="n_qubits", columns="n_layers", values="val_auc")
                sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[idx],
                            linewidths=0.5, cbar_kws={"label": "Best Val AUC"}, vmin=0.5, vmax=1.0)
                axes[idx].set_title(f"{title}\n(Val AUC)")
                axes[idx].set_xlabel("n_layers"); axes[idx].set_ylabel("n_qubits")
        plt.tight_layout()
        plt.savefig(out / "sweep_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        best_lr = df.groupby(["n_qubits","n_layers"])["val_auc"].max().reset_index()
        pivot   = best_lr.pivot(index="n_qubits", columns="n_layers", values="val_auc")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Best Val AUC"})
        ax.set_title("Hyperparameter Sweep — Val AUC\n(best LR per config)")
        ax.set_xlabel("n_layers"); ax.set_ylabel("n_qubits")
        plt.tight_layout()
        plt.savefig(out / "sweep_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  Sweep heatmap saved: {out / 'sweep_heatmap.png'}")


def select_top_sweep_configs(results: list, top_k: int = 3) -> list:
    """
    Select top sweep configurations strictly by validation AUC (and secondarily test metrics),
    ensuring global top performers (such as q6_l3 or q8_l3) are never filtered out.
    """
    if not results:
        return []
    # Deduplicate by configuration signature (params, qubits, layers, reupload)
    unique = []
    seen = set()
    for r in sorted(results, key=lambda x: (-x.get("best_val_auc", 0.0), -x.get("test_aupr", 0.0), -x.get("test_f1", 0.0))):
        key = (r["n_qubits"], r["n_layers"], r.get("reupload", True))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 9.  NOISE ROBUSTNESS (4)
# ══════════════════════════════════════════════════════════════════════════════

def run_noise_robustness(best_regime_A_result: dict) -> list:
    print("\n" + "═"*70)
    print("  NOISE ROBUSTNESS EVALUATION")
    print("═"*70)
    out = OUT_DIR / "noise"
    out.mkdir(parents=True, exist_ok=True)
    
    seed_everything(42)
    n_qubits = best_regime_A_result["n_qubits"]
    n_layers = best_regime_A_result["n_layers"]
    lr       = best_regime_A_result["lr"]

    X_train, y_train, X_val, y_val, _, _, _ = load_split(n_qubits, noise_sigma=0.0)
    train_dl, val_dl, _ = make_loaders(
        X_train, y_train, X_val, y_val, X_val, y_val,
        TRAIN_CFG["batch_size"]
    )
    clean_res = train_vqc(n_qubits, n_layers, lr, train_dl, val_dl,
                          out, label="[Noise baseline — clean train]", reupload=True)
    model = clean_res["model_ref"]
    model.load_state_dict(clean_res["best_state"])

    noise_results = []
    for sigma in NOISE_SIGMAS:
        _, _, _, _, X_test_n, y_test_n, _ = load_split(n_qubits, noise_sigma=sigma)
        test_dl_n = DataLoader(
            TensorDataset(
                torch.tensor(X_test_n, dtype=torch.float32),
                torch.tensor(y_test_n, dtype=torch.long)
            ),
            batch_size=TRAIN_CFG["batch_size"], shuffle=False
        )
        _, ts_acc, ts_f1, ts_auc, ts_aupr, _, _, _, ts_f1_opt, tau_n, _, _ = eval_epoch(model, test_dl_n, nn.BCELoss())
        print(f"  σ={sigma:.2f}: AUC={ts_auc:.4f}  AUPR={ts_aupr:.4f}  F1={ts_f1:.4f}  F1(τ*)={ts_f1_opt:.4f}")
        noise_results.append({
            "noise_sigma": sigma,
            "test_auc_roc": round(ts_auc, 4),
            "test_aupr": round(ts_aupr, 4),
            "test_f1": round(ts_f1, 4),
            "test_f1_opt": round(ts_f1_opt, 4),
            "opt_threshold": round(tau_n, 4),
            "test_accuracy": round(ts_acc, 4),
            "n_qubits": n_qubits,
            "n_layers": n_layers,
        })

    df_noise = pd.DataFrame(noise_results)
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.plot(df_noise["noise_sigma"], df_noise["test_auc_roc"],
            marker="o", lw=2, color="#4C72B0", label="HQCNN (VQC)")
    ax.set_xlabel("Gaussian Noise σ (injected into PCA features)")
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title(f"Robustness Under Scanner Noise\n"
                 f"HQCNN q={n_qubits} l={n_layers} — Mendeley Test Set")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.02])
    plt.tight_layout()
    plt.savefig(out / "noise_robustness_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Noise robustness curve saved.")

    df_noise.to_csv(out / "noise_results.csv", index=False, encoding="utf-8-sig")
    return noise_results


# ══════════════════════════════════════════════════════════════════════════════
# 10. RE-UPLOADING ABLATION, FINALIST EXPORT & REPEATED CV
# ══════════════════════════════════════════════════════════════════════════════

def run_reuploading_ablation(finalist_configs: list) -> list:
    """
    Tier 1 Item 6: Ablate data re-uploading on the finalist configurations.
    """
    print("\n" + "═"*70)
    print("  DATA RE-UPLOADING ABLATION (Finalist Configurations)")
    print("═"*70)
    out = OUT_DIR / "reupload_ablation"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    
    for cfg in finalist_configs:
        nq = cfg["n_qubits"]
        nl = cfg["n_layers"]
        lr = cfg.get("lr", 0.01)
        
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(nq)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )
        res_no = train_vqc(nq, nl, lr, train_dl, val_dl, out,
                           label="[No Re-uploading]", reupload=False)
        res_no["regime"] = f"A — ablation (no reupload, q={nq}, l={nl})"
        res_no = test_evaluate(res_no, test_dl, val_dl=val_dl)
        res_no["notes"] = "Data encoded once at layer 0"
        results.append(res_no)
        print(f"    No Re-upload (q={nq}, l={nl}): Val AUC={res_no['best_val_auc']:.4f} | Test AUC={res_no['test_auc_roc']:.4f} | Test F1(τ*)={res_no['test_f1_opt']:.4f}")
        
        # Mirror checkpoint to regime_A
        ckpt_name = f"vqc_q{nq}_l{nl}_lr{lr}_noreupload.pt"
        src_ckpt = out / ckpt_name
        if src_ckpt.exists():
            regime_a_dir = OUT_DIR / "regime_A"
            regime_a_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_ckpt, regime_a_dir / ckpt_name)
            gen_name = f"vqc_q{nq}_l{nl}_lr{lr}.pt"
            if not (regime_a_dir / gen_name).exists():
                shutil.copy2(src_ckpt, regime_a_dir / gen_name)
        
    return results


def select_finalists(sweep_results: list, top_k: int = 3, min_val_auc: float = 0.88) -> dict:
    """
    Select two explicitly labeled, non-exclusive categories of finalists:
      - performance_finalists: top_k strictly by val_auc (headline comparison)
      - efficiency_finalists: smallest parameter count that clears min_val_auc
    """
    def to_finalist_dict(r, rank_label):
        nq = int(r["n_qubits"])
        nl = int(r["n_layers"])
        lr = float(r["lr"])
        reup = bool(r.get("reupload", True))
        regime_str = r.get("regime", "A")
        regime_code = "B" if "B" in regime_str else "A"
        reup_tag = "" if reup else "_noreup"
        return {
            "name": f"HQCNN_{regime_code}_q{nq}_l{nl}{reup_tag}",
            "regime": regime_code,
            "n_qubits": nq,
            "n_layers": nl,
            "lr": lr,
            "reupload": reup,
            "val_auc": float(r.get("best_val_auc", r.get("val_auc", 0.0))),
            "test_auc": float(r.get("test_auc_roc", r.get("test_auc", 0.0))),
            "test_aupr": float(r.get("test_aupr", 0.0)),
            "vqc_params": int(r.get("vqc_params", nq * nl + 1)),
            "rank": rank_label,
            "notes": r.get("notes", ""),
        }

    # Rank by validation performance
    ranked_by_perf = sorted(sweep_results, key=lambda r: (
        -r.get("best_val_auc", r.get("val_auc", 0.0)),
        -r.get("test_aupr", 0.0),
        r.get("vqc_params", 999)
    ))
    perf_finalists = [to_finalist_dict(r, f"Performance Finalist #{i+1}") for i, r in enumerate(ranked_by_perf[:top_k])]

    # Rank by efficiency among those clearing min_val_auc
    eligible = [r for r in sweep_results if r.get("best_val_auc", r.get("val_auc", 0.0)) >= min_val_auc]
    if not eligible:
        # Fallback to all results if threshold is too strict
        eligible = sweep_results
    ranked_by_size = sorted(eligible, key=lambda r: (
        r.get("vqc_params", 999),
        -r.get("best_val_auc", r.get("val_auc", 0.0))
    ))
    # Deduplicate efficiency finalists by (qubits, layers, reupload)
    eff_unique = []
    seen_eff = set()
    for r in ranked_by_size:
        key = (r["n_qubits"], r["n_layers"], r.get("reupload", True))
        if key not in seen_eff:
            seen_eff.add(key)
            eff_unique.append(r)

    eff_finalists = [to_finalist_dict(r, f"Efficiency Finalist #{i+1}") for i, r in enumerate(eff_unique[:top_k])]

    return {
        "performance_finalists": perf_finalists,
        "efficiency_finalists": eff_finalists,
        "min_val_auc_threshold_used": float(min_val_auc),
    }


def export_finalist_configs(regime_A_results: list, regime_B_results: list, sweep_results: list,
                            baseline_val_auc: float = 0.98) -> dict:
    """
    Tier 1 Item 2 Stage A & Priority 0:
    Select performance and efficiency finalists and export structured finalist_configs.json.
    """
    all_vqc_runs = []
    for r in regime_A_results + sweep_results:
        all_vqc_runs.append(r)

    # Adaptive minimum validation AUC for efficiency finalists (e.g. classical baseline val AUC - 0.10)
    adaptive_min_val_auc = max(0.85, baseline_val_auc - 0.10)

    finalist_dict = select_finalists(all_vqc_runs, top_k=3, min_val_auc=adaptive_min_val_auc)

    # If Regime B exists, add top Regime B to performance finalists if not already present
    if regime_B_results:
        best_B = max(regime_B_results, key=lambda r: r.get("best_val_auc", 0.0))
        finalist_dict["regime_b_finalist"] = {
            "name": f"HQCNN_B_q{best_B['n_qubits']}_l{best_B['n_layers']}",
            "regime": "B",
            "n_qubits": int(best_B["n_qubits"]),
            "n_layers": int(best_B["n_layers"]),
            "lr": float(best_B["lr"]),
            "reupload": True,
            "val_auc": float(best_B.get("best_val_auc", 0.0)),
            "test_auc": float(best_B.get("test_auc_roc", 0.0)),
            "vqc_params": int(best_B.get("vqc_params", best_B["n_qubits"] * best_B["n_layers"] + 1)),
            "rank": "Regime B Primary Finalist",
        }

    # Ensure all finalist checkpoints are mirrored into regime_A for seamless downstream access
    regime_a_dir = OUT_DIR / "regime_A"
    regime_a_dir.mkdir(parents=True, exist_ok=True)
    all_finalists = list(finalist_dict.get("performance_finalists", [])) + list(finalist_dict.get("efficiency_finalists", []))
    if "regime_b_finalist" in finalist_dict:
        all_finalists.append(finalist_dict["regime_b_finalist"])

    for f_cfg in all_finalists:
        nq = int(f_cfg["n_qubits"])
        nl = int(f_cfg["n_layers"])
        lr = float(f_cfg.get("lr", 0.01))
        reup = bool(f_cfg.get("reupload", True))
        suffix = "" if reup else "_noreupload"
        ckpt_name = f"vqc_q{nq}_l{nl}_lr{lr}{suffix}.pt"
        gen_name = f"vqc_q{nq}_l{nl}_lr{lr}.pt"

        for src_dir in [OUT_DIR / "sweep", OUT_DIR / "reupload_ablation", OUT_DIR / "regime_B", regime_a_dir]:
            src = src_dir / ckpt_name
            if src.exists():
                if src.parent != regime_a_dir:
                    shutil.copy2(src, regime_a_dir / ckpt_name)
                    print(f"  [Finalist Sync] Mirrored {ckpt_name} to regime_A")
                if not (regime_a_dir / gen_name).exists():
                    shutil.copy2(src, regime_a_dir / gen_name)
                break

    finalist_path = OUT_DIR / "finalist_configs.json"
    with open(finalist_path, "w") as f:
        json.dump(finalist_dict, f, indent=2)
    print(f"\n  ✓ Exported finalist configurations to: {finalist_path}")
    print(f"    Performance finalists: {[f['name'] for f in finalist_dict['performance_finalists']]}")
    print(f"    Efficiency finalists:  {[f['name'] for f in finalist_dict['efficiency_finalists']]}")
    return finalist_dict


def run_repeated_cv(finalist_configs: Union[list, dict], n_splits: int = 5, n_repeats: int = 3) -> dict:
    """
    Tier 1 Items 1 & 2 Stages B & C, Priority 0 & 5:
    Repeated Stratified CV for controls and the union of performance + efficiency finalists.
    Logs explicit TrainableParams column in cv_summary.csv.
    """
    print("\n" + "═"*70)
    print(f"  CROSS-VALIDATION EVALUATION ({n_splits}-fold × {n_repeats}-repeat = {n_splits*n_repeats} folds)")
    print("═"*70)
    cv_dir = OUT_DIR / "cv_outputs"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    y_all = np.concatenate([
        np.load(FEAT_DIR / "labels_train.npy"),
        np.load(FEAT_DIR / "labels_val.npy"),
        np.load(FEAT_DIR / "labels_test.npy"),
    ])
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    cv_results = {}

    # Stage B: MicroMLP controls
    for dim in [4, 6]:
        f_tr = np.load(FEAT_DIR / f"features_train_pca{dim}.npy")
        f_va = np.load(FEAT_DIR / f"features_val_pca{dim}.npy")
        f_te = np.load(FEAT_DIR / f"features_test_pca{dim}.npy")
        X_all = np.vstack([f_tr, f_va, f_te])
        
        fold_aucs, fold_auprs, fold_f1s = [], [], []
        mlp_sample = ClassicalMicroMLP(dim, hidden_dim=2)
        n_params = mlp_sample.count_params()
        for fold, (tr_idx, te_idx) in enumerate(rskf.split(X_all, y_all)):
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_te, y_te = X_all[te_idx], y_all[te_idx]
            
            sc = MinMaxScaler(feature_range=(0, 1)).fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_te_s = sc.transform(X_te)
            
            mlp = ClassicalMicroMLP(dim, hidden_dim=2)
            opt = optim.Adam(mlp.parameters(), lr=0.01)
            crit = nn.BCELoss()
            ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=16, shuffle=True)
            mlp.train()
            for _ in range(15):
                for X_b, y_b in loader:
                    opt.zero_grad()
                    loss = crit(mlp(X_b), y_b)
                    loss.backward()
                    opt.step()
            mlp.eval()
            with torch.no_grad():
                probs = mlp(torch.tensor(X_te_s, dtype=torch.float32)).numpy()
            
            fold_auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else 0.0
            fold_aupr = average_precision_score(y_te, probs) if len(set(y_te)) > 1 else 0.0
            tau = find_optimal_threshold_youden(y_te, probs)
            fold_f1 = f1_score(y_te, (probs > tau).astype(int), zero_division=0)
            
            fold_aucs.append(fold_auc)
            fold_auprs.append(fold_aupr)
            fold_f1s.append(fold_f1)
            
        lbl = f"Micro-MLP (dim={dim})"
        cv_results[lbl] = {
            "model": lbl,
            "trainable_params": n_params,
            "auc_mean": round(float(np.mean(fold_aucs)), 4),
            "auc_std": round(float(np.std(fold_aucs)), 4),
            "aupr_mean": round(float(np.mean(fold_auprs)), 4),
            "aupr_std": round(float(np.std(fold_auprs)), 4),
            "f1_mean": round(float(np.mean(fold_f1s)), 4),
            "f1_std": round(float(np.std(fold_f1s)), 4),
            "fold_aucs": [round(x, 4) for x in fold_aucs],
        }
        print(f"  CV [Stage B] {lbl:25s} ({n_params} params) | AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f} | PR-AUC: {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")

    # Stage C: Union of Performance & Efficiency Finalists (de-duplicated)
    eval_configs = []
    if isinstance(finalist_configs, dict):
        eval_configs = finalist_configs.get("performance_finalists", []) + finalist_configs.get("efficiency_finalists", [])
    elif isinstance(finalist_configs, list):
        eval_configs = finalist_configs

    unique_configs = []
    seen_keys = set()
    for cfg in eval_configs:
        key = (cfg["n_qubits"], cfg["n_layers"], cfg.get("reupload", True))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_configs.append(cfg)

    for cfg in unique_configs:
        nq = cfg["n_qubits"]
        nl = cfg["n_layers"]
        lr = cfg.get("lr", 0.01)
        reup = cfg.get("reupload", True)
        name = cfg.get("name", f"HQCNN_q{nq}_l{nl}")
        vqc_params = int(cfg.get("vqc_params", nq * nl + 1))
        
        f_tr = np.load(FEAT_DIR / f"features_train_pca{nq}.npy")
        f_va = np.load(FEAT_DIR / f"features_val_pca{nq}.npy")
        f_te = np.load(FEAT_DIR / f"features_test_pca{nq}.npy")
        X_all = np.vstack([f_tr, f_va, f_te])
        
        fold_aucs, fold_auprs, fold_f1s = [], [], []
        for fold, (tr_idx, te_idx) in enumerate(rskf.split(X_all, y_all)):
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_te, y_te = X_all[te_idx], y_all[te_idx]
            
            sc = MinMaxScaler(feature_range=(0, 1)).fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_te_s = sc.transform(X_te)
            
            vqc = HQCNNClassifier(nq, nl, reupload=reup)
            opt = optim.Adam(vqc.parameters(), lr=lr)
            crit = nn.BCELoss()
            ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            vqc.train()
            for _ in range(12):
                for X_b, y_b in loader:
                    opt.zero_grad()
                    loss = crit(vqc(X_b), y_b)
                    loss.backward()
                    opt.step()
            vqc.eval()
            with torch.no_grad():
                probs = vqc(torch.tensor(X_te_s, dtype=torch.float32)).numpy()
                
            fold_auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else 0.0
            fold_aupr = average_precision_score(y_te, probs) if len(set(y_te)) > 1 else 0.0
            tau = find_optimal_threshold_youden(y_te, probs)
            fold_f1 = f1_score(y_te, (probs > tau).astype(int), zero_division=0)
            
            fold_aucs.append(fold_auc)
            fold_auprs.append(fold_aupr)
            fold_f1s.append(fold_f1)
            
        cv_results[name] = {
            "model": name,
            "trainable_params": vqc_params,
            "auc_mean": round(float(np.mean(fold_aucs)), 4),
            "auc_std": round(float(np.std(fold_aucs)), 4),
            "aupr_mean": round(float(np.mean(fold_auprs)), 4),
            "aupr_std": round(float(np.std(fold_auprs)), 4),
            "f1_mean": round(float(np.mean(fold_f1s)), 4),
            "f1_std": round(float(np.std(fold_f1s)), 4),
            "fold_aucs": [round(x, 4) for x in fold_aucs],
        }
        print(f"  CV [Stage C] {name:25s} ({vqc_params} params) | AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f} | PR-AUC: {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")

    with open(OUT_DIR / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    pd.DataFrame(list(cv_results.values())).to_csv(OUT_DIR / "cv_summary.csv", index=False)
    print(f"  ✓ Saved CV summary with parameter counts: {OUT_DIR / 'cv_summary.csv'}")
    return cv_results


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print("  3–5 — VQC TRAINING, SWEEP & ABLATION")
    print("  QFL Breast Cancer Classification | HQCNN Pipeline")
    print("═"*70)

    from cache_check import already_done, CACHE
    if already_done("vqc"):
        print("  [SKIP] VQC pipeline already completed (cache hit: vqc)")
        return

    all_results = []
    seed_everything(42)

    # ── 3_: Regime A ────────────────────────────────────────────────────
    regime_A_results = run_regime_A()
    all_results.extend(regime_A_results)

    # ── 4a_: Regime B ───────────────────────────────────────────────────
    regime_B_results = run_regime_B()
    all_results.extend(regime_B_results)

    # ── 5_: Hyperparameter sweep ────────────────────────────────────────
    sweep_results = run_sweep()
    for r in sweep_results:
        r["notes"] = f"sweep lr={r['lr']}"
    all_results.extend(sweep_results)

    # ── Export finalist configurations (Priority 0) ──────────────────────
    baseline_val_auc = 0.98
    if BASELINE_JSON.exists():
        try:
            with open(BASELINE_JSON) as f:
                baseline_val_auc = float(json.load(f).get("best_val_auc", 0.98))
        except Exception:
            pass

    finalists = export_finalist_configs(regime_A_results, regime_B_results, sweep_results,
                                        baseline_val_auc=baseline_val_auc)

    # ── 4b_: Noise robustness on Top Performance Finalist ────────────────
    perf_finalists = finalists.get("performance_finalists", [])
    primary_finalist = perf_finalists[0] if perf_finalists else max(regime_A_results + sweep_results, key=lambda r: r.get("best_val_auc", 0.0))
    print(f"\n  Running noise robustness on Primary Performance Finalist: {primary_finalist['name']} "
          f"(q={primary_finalist['n_qubits']}, l={primary_finalist['n_layers']}, lr={primary_finalist['lr']})")
    noise_results = run_noise_robustness(primary_finalist)
    for nr in noise_results:
        if nr["noise_sigma"] > 0:
            nr["regime"]  = "A — noise robustness"
            nr["vqc_params"] = primary_finalist.get("vqc_params", primary_finalist["n_qubits"] * primary_finalist["n_layers"] + 1)
            nr["notes"]   = f"σ={nr['noise_sigma']}"
            nr["best_val_auc"] = primary_finalist.get("val_auc", primary_finalist.get("best_val_auc", 0.0))
            all_results.append(nr)

    # ── Data re-uploading ablation on Top Finalists ──────────────────────
    eval_ablation_configs = (finalists.get("performance_finalists", [])[:1] +
                             finalists.get("efficiency_finalists", [])[:1])
    reupload_ablation_results = run_reuploading_ablation(eval_ablation_configs)
    all_results.extend(reupload_ablation_results)

    # ── Classical control baselines ──────────────────────────────────────────
    classical_results = run_classical_control_configs(n_features=4)
    all_results.extend(classical_results)

    # ── Classical baselines matched to top VQC sweep budgets --------------
    top_vqc_runs = select_top_sweep_configs(sweep_results, top_k=3)
    if top_vqc_runs:
        print("\n  Running parameter-matched classical MLP baselines for top sweep configs...")
        matched_results = run_parameter_matched_mlp_baselines(top_vqc_runs)
        all_results.extend(matched_results)

    # ── Repeated Stratified CV scheme (Tier 1 Items 1 & 2 Stages B & C) ─
    run_repeated_cv(finalists)

    # ── Build & save ablation table ──────────────────────────────────────────
    print("\n" + "═"*70)
    print("  BUILDING ABLATION TABLE")
    print("═"*70)
    ablation_df = build_ablation_table(all_results, BASELINE_JSON)
    ablation_df.to_csv(OUT_DIR / "ablation_table.csv", index=False, encoding="utf-8-sig")
    plot_ablation_table(ablation_df, OUT_DIR / "ablation_table.png")

    print("\n━"*70)
    print("  ABLATION TABLE SUMMARY")
    print("━"*70)
    display_cols = ["Model","Regime","Qubits","Layers","TrainableParams",
                    "NoiseSigma","TestAUC","TestAUCPR","TestF1"]
    print(ablation_df[display_cols].to_string(index=False))

    CACHE.mark_done("vqc")
    print(f"\n✓ 3–5 COMPLETE. All outputs in: {OUT_DIR}")
    print("  Next step → 6_7_uq.py (Uncertainty Quantification)")


if __name__ == "__main__":
    main()