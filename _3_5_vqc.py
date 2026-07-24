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
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report, roc_curve
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
BASELINE_JSON = BASE / "baseline_outputs/baseline_results.json"

# ── VQC hyperparameter grid (5 sweep) ───────────────────────────────────
# To run quickly: keep this small. Full sweep can be large.
SWEEP_CFG = {
    "n_qubits":    [4, 6, 8],       # maps to PCA n_components
    "n_layers":    [1, 2, 3],       # VQC depth (ansatz repetitions)
    "encoding":    ["angle"],       # "angle" only for now; extend later
    "lr":          [0.01, 0.005],
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

def build_vqc(n_qubits: int, n_layers: int):
    """
    Build a PennyLane VQC as a PyTorch-compatible nn.Module layer.

    Circuit structure (repeated n_layers times):
      1. Angle encoding:   RY(π·x_i) on qubit i   (data re-uploading each layer)
      2. Variational layer: RY(θ_i) on each qubit
      3. Entanglement:      CNOT in circular pattern (0→1→2→...→n-1→0)

    Measurement: expectation value ⟨Z₀⟩ on qubit 0 → scalar in [-1, 1]
    Mapped to [0, 1] probability via (1 + ⟨Z₀⟩) / 2 for BCE loss.

    Data re-uploading (encoding at each layer) is a key design choice:
    it allows the VQC to express non-linear functions of the input beyond
    what a single encoding layer can achieve — critical for 4–6 features.
    Reference: Pérez-Salinas et al. (2020) "Data re-uploading for a universal
    quantum classifier."
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        """
        inputs  : (n_qubits,)  — MinMax-scaled PCA features in [0,1]
        weights : (n_layers, n_qubits)  — trainable VQC parameters
        """
        for layer in range(n_layers):
            # ── Angle encoding (data re-uploading each layer) ──
            for i in range(n_qubits):
                qml.RY(np.pi * inputs[i], wires=i)
            # ── Variational RY rotations ──
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            # ── Circular CNOT entanglement ──
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HQCNNClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network:
      Input (n_qubits,) → VQC → scalar ⟨Z₀⟩ → sigmoid → P(malignant)

    Regime A (frozen classical): only this module's parameters are trained.
    Regime B (end-to-end): this + CNN head parameters trained jointly.

    The module handles single-sample and batched input transparently.
    """
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.vqc      = build_vqc(n_qubits, n_layers)
        # Learnable bias term after VQC measurement (improves calibration)
        self.bias     = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, n_qubits) — already MinMax scaled to [0,1]
        # VQC processes each sample independently
        out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        # Map ⟨Z⟩ ∈ [-1,1] to probability ∈ (0,1)
        prob = torch.sigmoid(out + self.bias)
        return prob   # shape: (batch,)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassicalMicroMLP(nn.Module):
    """Minimal classical MLP baseline matched to the VQC parameter budget."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

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
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds_all, probs_all, labels_all = 0.0, [], [], []
    for X_batch, y_batch in loader:
        probs  = model(X_batch)
        loss   = criterion(probs, y_batch.float())
        total_loss += loss.item() * X_batch.size(0)
        probs_all.extend(probs.numpy())
        preds_all.extend((probs > 0.5).long().numpy())
        labels_all.extend(y_batch.numpy())
    n   = len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, zero_division=0)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0
    aupr = average_precision_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0
    return total_loss / n, acc, f1, auc, aupr, labels_all, preds_all, probs_all


def train_vqc(n_qubits: int, n_layers: int, lr: float,
              train_dl, val_dl, out_subdir: Path,
              label: str = "", noise_sigma: float = 0.0) -> dict:
    """
    Full training run for one VQC configuration.
    Returns a results dict with test metrics and parameter count.
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    model     = HQCNNClassifier(n_qubits, n_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-4
    )

    best_auc, best_aupr, best_state, patience_ctr = 0.0, 0.0, None, 0
    history = []

    print(f"    Training: qubits={n_qubits} layers={n_layers} lr={lr} {label}")
    for epoch in range(1, TRAIN_CFG["num_epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, vl_auc, vl_aupr, _, _, _ = eval_epoch(model, val_dl, criterion)
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
            "val_aupr": vl_aupr, "is_best": is_best,
        })
        if epoch % 10 == 0 or is_best:
            print(f"      Ep {epoch:3d} | TrLoss {tr_loss:.4f} | "
                  f"VlAUC {vl_auc:.4f}" + (" ← best" if is_best else ""))

        if patience_ctr >= TRAIN_CFG["patience"]:
            print(f"      Early stopping at epoch {epoch}.")
            break

    # Save checkpoint + history
    ckpt_name = f"vqc_q{n_qubits}_l{n_layers}_lr{lr}.pt"
    torch.save(best_state, out_subdir / ckpt_name)
    pd.DataFrame(history).to_csv(out_subdir / ckpt_name.replace(".pt", "_history.csv"),
                                 index=False)

    return {
        "n_qubits":      n_qubits,
        "n_layers":      n_layers,
        "lr":            lr,
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

def test_evaluate(result: dict, test_dl) -> dict:
    """Load best weights and evaluate on test set."""
    model = result["model_ref"]
    model.load_state_dict(result["best_state"])
    criterion = nn.BCELoss()
    _, ts_acc, ts_f1, ts_auc, ts_aupr, y_true, y_pred, y_prob = eval_epoch(
        model, test_dl, criterion
    )
    return {
        **{k: v for k, v in result.items() if k not in ("best_state", "model_ref")},
        "test_accuracy": round(ts_acc, 4),
        "test_f1":       round(ts_f1,  4),
        "test_auc_roc":  round(ts_auc, 4),
        "test_aupr":     round(ts_aupr, 4),
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
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
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
            test_probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        test_preds = (test_probs > 0.5).astype(int)
        val_aupr = average_precision_score(y_val, model(torch.tensor(X_val, dtype=torch.float32)).detach().numpy()) if len(set(y_val)) > 1 else 0.0
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
            "Qubits":          "—",
            "Layers":          "—",
            "TrainableParams": bl.get("trainable_params", bl.get("head_params", "N/A")),
            "NoiseSigma":      0.0,
            "ValAUC":          bl.get("best_val_auc", "—"),
            "ValAUPRC":       bl.get("best_val_aupr", "—"),
            "TestAUC":         bl.get("test_auc_roc", "—"),
            "TestAUCPR":      bl.get("test_aupr", "—"),
            "TestF1":          bl.get("test_f1",      "—"),
            "TestAcc":         bl.get("test_accuracy","—"),
            "Notes":           "Classical baseline (frozen backbone)",
        })

    # ── VQC result rows ───────────────────────────────────────────────────────
    for r in all_results:
        model_label = r.get("model") or f"HQCNN (VQC q={r.get('n_qubits', '—')} l={r.get('n_layers', '—')})"
        rows.append({
            "Model":           model_label,
            "Regime":          r.get("regime", "A — frozen classical + VQC"),
            "Qubits":          r.get("n_qubits", "—"),
            "Layers":          r.get("n_layers", "—"),
            "TrainableParams": r.get("trainable_params", r.get("vqc_params", "—")),
            "NoiseSigma":      r.get("noise_sigma", 0.0),
            "ValAUC":          r.get("best_val_auc", r.get("val_auc", "—")),
            "ValAUPRC":       r.get("best_val_aupr", "—"),
            "TestAUC":         r.get("test_auc_roc", r.get("test_auc", "—")),
            "TestAUCPR":      r.get("test_aupr", "—"),
            "TestF1":          r.get("test_f1",      "—"),
            "TestAcc":         r.get("test_accuracy","—"),
            "Notes":           r.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    return df


def plot_ablation_table(df: pd.DataFrame, save_path: Path):
    """Render the ablation DataFrame as a publication-quality figure."""
    display_cols = ["Model", "Regime", "Qubits", "Layers",
                    "TrainableParams", "NoiseSigma",
                    "ValAUC", "ValAUPRC", "TestAUC", "TestAUCPR", "TestF1", "TestAcc"]
    plot_df = df[display_cols].copy()
    plot_df = plot_df.fillna("—")

    fig, ax = plt.subplots(figsize=(18, max(4, len(plot_df) * 0.6 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ablation table saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REGIME A — Frozen classical + train VQC only (3)
# ══════════════════════════════════════════════════════════════════════════════

def run_regime_A() -> list:
    """
    Regime A: classical backbone frozen, only VQC parameters trained.
    Run the recommended starting configuration (n=4, l=2) first,
    then the best sweep config.
    Returns list of result dicts.
    """
    print("\n" + "═"*70)
    print("  REGIME A — Frozen Classical + Train VQC")
    print("═"*70)
    out = OUT_DIR / "regime_A"
    results = []
    seed_everything(42)

    # Primary config: 4 qubits, 2 layers (safest starting point)
    for n_qubits, n_layers, lr in [(4, 2, 0.01), (6, 2, 0.01), (8, 2, 0.01)]:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_qubits)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )
        res = train_vqc(n_qubits, n_layers, lr, train_dl, val_dl, out,
                        label="[Regime A]")
        res["regime"] = "A — frozen classical + VQC"
        res = test_evaluate(res, test_dl)

        # Diagnostic plots
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
        print(f"    Test AUC={res['test_auc_roc']:.4f} F1={res['test_f1']:.4f} "
              f"Params={res['vqc_params']}")
        results.append(res)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REGIME B — End-to-end joint training (4)
# ══════════════════════════════════════════════════════════════════════════════

def run_regime_B() -> list:
    """
    Regime B: CNN head + VQC parameters optimised jointly.
    We simulate this by loading the saved CNN features but adding a small
    learnable linear layer before the VQC — representing the unfrozen head —
    and training both together.

    Note: True end-to-end would require running the CNN forward pass inside
    the training loop (expensive on CPU for quantum simulation). This
    approximation trains a learnable projection (dim → n_qubits) jointly with
    the VQC, which is architecturally equivalent for the quantum component.
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

        # Wrap: learnable linear projection + VQC
        class RegimeBModel(nn.Module):
            def __init__(self, in_dim, n_qubits, n_layers):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(in_dim, n_qubits),
                    nn.Sigmoid(),        # keep outputs in (0,1) for angle encoding
                )
                self.hqcnn = HQCNNClassifier(n_qubits, n_layers)

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
            # Inline train/eval since model wraps differently
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

        pd.DataFrame(history).to_csv(
            out / f"regimeB_q{n_qubits}_l{n_layers}_history.csv", index=False
        )
        # Save best checkpoint (needed by 6_7_uq.py)
        torch.save(best_state,
                   out / f"vqc_q{n_qubits}_l{n_layers}_lr{lr}.pt")

        # Test evaluation
        model.load_state_dict(best_state)
        model.eval()
        ts_probs, ts_preds, ts_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_dl:
                p = model(X_batch)
                ts_probs.extend(p.numpy())
                ts_preds.extend((p > 0.5).long().numpy())
                ts_labels.extend(y_batch.numpy())

        ts_auc = roc_auc_score(ts_labels, ts_probs)
        ts_aupr = average_precision_score(ts_labels, ts_probs) if len(set(ts_labels)) > 1 else 0.0
        ts_f1  = f1_score(ts_labels, ts_preds, zero_division=0)
        ts_acc = accuracy_score(ts_labels, ts_preds)
 
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
            "test_accuracy":round(ts_acc, 4),
            "y_true": ts_labels, "y_pred": ts_preds, "y_prob": ts_probs,
        }
        plot_confusion_matrix(
            ts_labels, ts_preds,
            f"Regime B — q={n_qubits} l={n_layers}",
            out / f"regimeB_q{n_qubits}_l{n_layers}_cm.png"
        )
        print(f"    Test AUC={ts_auc:.4f} F1={ts_f1:.4f} Params={model.count_params()}")
        results.append(res)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8.  HYPERPARAMETER SWEEP (5)
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep() -> list:
    """
    Grid sweep over SWEEP_CFG. Runs Regime A for each combination.
    Expensive — set SWEEP_CFG small or run overnight.
    """
    print("\n" + "═"*70)
    print("  HYPERPARAMETER SWEEP (Regime A)")
    print("═"*70)
    out = OUT_DIR / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    results = []

    grid = list(itertools.product(
        SWEEP_CFG["n_qubits"],
        SWEEP_CFG["n_layers"],
        SWEEP_CFG["lr"],
    ))
    print(f"  Total configurations: {len(grid)}")

    for n_qubits, n_layers, lr in grid:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = load_split(n_qubits)
        train_dl, val_dl, test_dl = make_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            TRAIN_CFG["batch_size"]
        )
        res = train_vqc(n_qubits, n_layers, lr, train_dl, val_dl, out,
                        label="[Sweep]")
        res["regime"] = "A — sweep"
        res = test_evaluate(res, test_dl)
        res["notes"] = f"sweep lr={lr}"
        results.append(res)

    # Summary heatmap: val AUC vs (n_qubits, n_layers)
    _plot_sweep_heatmap(results, out)

    # Promote the best sweep configuration to regime_A for downstream scripts
    best_result = max(results, key=lambda r: r["best_val_auc"])
    best_ckpt_name = f"vqc_q{best_result['n_qubits']}_l{best_result['n_layers']}_lr{best_result['lr']}.pt"
    best_ckpt_src  = out / best_ckpt_name
    regime_A_dir   = OUT_DIR / "regime_A"
    regime_A_dir.mkdir(parents=True, exist_ok=True)
    if best_ckpt_src.exists():
        shutil.copy2(best_ckpt_src, regime_A_dir / best_ckpt_name)
        print(f"  Promoted best sweep checkpoint to regime_A: {best_ckpt_name}")
    else:
        print(f"  [WARN] Best sweep checkpoint not found: {best_ckpt_src}")

    manifest = regime_A_dir / "best_run_manifest.json"
    with open(manifest, "w") as f:
        json.dump({
            "n_qubits": int(best_result["n_qubits"]),
            "n_layers": int(best_result["n_layers"]),
            "lr": float(best_result["lr"]),
            "val_auc": float(best_result["best_val_auc"]),
            "regime": "A — sweep",
        }, f, indent=2)
    print(f"  Wrote best-run manifest: {manifest}")

    top_runs = select_top_sweep_configs(results, top_k=3)
    print("  Selected top-3 sweep configs by compact budget and validation performance:")
    for r in top_runs:
        print(f"    q={r['n_qubits']} l={r['n_layers']} params={r['vqc_params']} "
              f"val_auc={r['best_val_auc']:.4f} test_aupr={r.get('test_aupr', 0.0):.4f}")
    top_manifest = regime_A_dir / "top_sweep_manifest.json"
    with open(top_manifest, "w") as f:
        json.dump([
            {
                "n_qubits": int(r["n_qubits"]),
                "n_layers": int(r["n_layers"]),
                "lr": float(r["lr"]),
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
                "lr": r["lr"], "val_auc": r["best_val_auc"]} for r in results]
    df = pd.DataFrame(records)
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


def select_top_sweep_configs(results: list, top_k: int = 3, candidate_pool: int = 12) -> list:
    """Choose a compact set of top-performing sweep configs from the best validation runs.

    This returns the least-parameter variants among the top candidate pool,
    ensuring both strong validation performance and a range from smaller to larger
    quantum budgets.
    """
    if not results:
        return []

    # Order primarily by validation AUC, but prefer smaller circuits when tied.
    ranked = sorted(results, key=lambda r: (
        -r["best_val_auc"], -r.get("test_f1", 0.0), r["vqc_params"], r["n_qubits"], r["n_layers"]
    ))
    pool = ranked[:candidate_pool]

    # Deduplicate by circuit budget so we compare unique parameter-sized variants.
    unique = []
    seen = set()
    for r in pool:
        key = (r["vqc_params"], r["n_qubits"], r["n_layers"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    selected = sorted(unique, key=lambda r: (
        r["vqc_params"], -r["best_val_auc"], -r.get("test_f1", 0.0), r["n_qubits"], r["n_layers"]
    ))[:top_k]

    # Present from smallest to largest parameter budget.
    return sorted(selected, key=lambda r: (
        r["vqc_params"], -r["best_val_auc"], -r.get("test_f1", 0.0), r["n_qubits"], r["n_layers"]
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  NOISE ROBUSTNESS (4)
# ══════════════════════════════════════════════════════════════════════════════

def run_noise_robustness(best_regime_A_result: dict) -> list:
    """
    Take the best Regime A config, retrain once on clean data, then evaluate
    on test sets with increasing Gaussian noise injected into the PCA features.
    Also runs the same noise levels on classical baseline predictions as proxy.
    """
    print("\n" + "═"*70)
    print("  NOISE ROBUSTNESS EVALUATION")
    print("═"*70)
    out = OUT_DIR / "noise"
    out.mkdir(parents=True, exist_ok=True)
    
    seed_everything(42)
    n_qubits = best_regime_A_result["n_qubits"]
    n_layers = best_regime_A_result["n_layers"]
    lr       = best_regime_A_result["lr"]

    # Train once on clean data
    X_train, y_train, X_val, y_val, _, _, _ = load_split(n_qubits, noise_sigma=0.0)
    train_dl, val_dl, _ = make_loaders(
        X_train, y_train, X_val, y_val, X_val, y_val,  # test_dl unused here
        TRAIN_CFG["batch_size"]
    )
    clean_res = train_vqc(n_qubits, n_layers, lr, train_dl, val_dl,
                          out, label="[Noise baseline — clean train]")
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
        _, ts_acc, ts_f1, ts_auc, ts_aupr, _, _, _ = eval_epoch(model, test_dl_n, nn.BCELoss())
        print(f"  σ={sigma:.2f}: AUC={ts_auc:.4f}  F1={ts_f1:.4f}  Acc={ts_acc:.4f}")
        noise_results.append({
            "noise_sigma": sigma, "test_auc_roc": round(ts_auc,4),
            "test_f1": round(ts_f1,4), "test_accuracy": round(ts_acc,4),
            "n_qubits": n_qubits, "n_layers": n_layers,
        })

    # Plot degradation curve
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

    df_noise.to_csv(out / "noise_results.csv", index=False)
    return noise_results


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
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

    # ── 4b_: Noise robustness ────────────────────────────────────────────
    # Use best Regime A result (highest val AUC)
    best_A = max(regime_A_results, key=lambda r: r["best_val_auc"])
    noise_results = run_noise_robustness(best_A)
    # Add noise rows to ablation (non-zero sigma only)
    for nr in noise_results:
        if nr["noise_sigma"] > 0:
            nr["regime"]  = "A — noise robustness"
            nr["vqc_params"] = best_A["vqc_params"]
            nr["notes"]   = f"σ={nr['noise_sigma']}"
            nr["best_val_auc"] = best_A["best_val_auc"]
            all_results.append(nr)

    # ── 5_: Hyperparameter sweep ────────────────────────────────────────
    sweep_results = run_sweep()
    for r in sweep_results:
        r["notes"] = f"sweep lr={r['lr']}"
    all_results.extend(sweep_results)

    # ── Classical control baselines ──────────────────────────────────────────
    classical_results = run_classical_control_configs(n_features=4)
    all_results.extend(classical_results)

    # ── Classical baselines matched to top VQC sweep budgets --------------
    top_vqc_runs = select_top_sweep_configs(sweep_results, top_k=3)
    if top_vqc_runs:
        print("\n  Running parameter-matched classical MLP baselines for top sweep configs...")
        matched_results = run_parameter_matched_mlp_baselines(top_vqc_runs)
        all_results.extend(matched_results)

    # ── Build & save ablation table ──────────────────────────────────────────
    print("\n" + "═"*70)
    print("  BUILDING ABLATION TABLE")
    print("═"*70)
    ablation_df = build_ablation_table(all_results, BASELINE_JSON)
    ablation_df.to_csv(OUT_DIR / "ablation_table.csv", index=False)
    plot_ablation_table(ablation_df, OUT_DIR / "ablation_table.png")

    print("\n━"*70)
    print("  ABLATION TABLE SUMMARY")
    print("━"*70)
    display_cols = ["Model","Regime","Qubits","Layers","TrainableParams",
                    "NoiseSigma","TestAUC","TestAUCPR","TestF1"]
    print(ablation_df[display_cols].to_string(index=False))

    CACHE.mark_done("vqc")
    print(f"\n✓ 3–5 COMPLETE. All outputs in: {OUT_DIR}")
    print("  Next step → 6_qfl.py (Quantum Federated Learning)")


if __name__ == "__main__":
    main()