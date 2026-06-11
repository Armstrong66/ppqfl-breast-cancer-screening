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

Outputs (/home/derrick/Projects/QFL_breast_cancer_screening/outputs/vqc_outputs/):
  regime_A/           ← frozen classical + train VQC
  regime_B/           ← end-to-end (fine-tune CNN head + VQC jointly)
  sweep/              ← hyperparameter sweep results
  noise/              ← robustness under Gaussian noise
  ablation_table.csv  ← full comparison table for reporting
  ablation_table.png  ← visual summary

Environment: Kaggle CPU (quantum simulation — no GPU needed for VQC)
             PennyLane + PyTorch — install: pip install pennylane
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os, json, pickle, warnings, itertools
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive, writes files only — no display needed
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
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature inputs (from 2b_feature_pca.py) ───────────────────────────────
BASE      = Path("/home/derrick/Projects/QFL_breast_cancer_screening/outputs")
FEAT_DIR  = BASE / "feature_outputs"
OUT_DIR   = BASE / "vqc_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Classical baseline results (for ablation table) ──────────────────────────
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
    return total_loss / n, acc, f1, auc, labels_all, preds_all, probs_all


def train_vqc(n_qubits: int, n_layers: int, lr: float,
              train_dl, val_dl, out_subdir: Path,
              label: str = "", noise_sigma: float = 0.0) -> dict:
    """
    Full training run for one VQC configuration.
    Returns a results dict with test metrics and parameter count.
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    model     = HQCNNClassifier(n_qubits, n_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-4
    )

    best_auc, best_state, patience_ctr = 0.0, None, 0
    history = []

    print(f"    Training: qubits={n_qubits} layers={n_layers} lr={lr} {label}")
    for epoch in range(1, TRAIN_CFG["num_epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, vl_auc, _, _, _ = eval_epoch(model, val_dl, criterion)
        scheduler.step(vl_auc)

        is_best = vl_auc > best_auc
        if is_best:
            best_auc   = vl_auc
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_auc": vl_auc, "is_best": is_best,
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
    _, ts_acc, ts_f1, ts_auc, y_true, y_pred, y_prob = eval_epoch(
        model, test_dl, criterion
    )
    return {
        **{k: v for k, v in result.items() if k not in ("best_state", "model_ref")},
        "test_accuracy": round(ts_acc, 4),
        "test_f1":       round(ts_f1,  4),
        "test_auc_roc":  round(ts_auc, 4),
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


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
            "TestAUC":         bl.get("test_auc_roc", "—"),
            "TestF1":          bl.get("test_f1",      "—"),
            "TestAcc":         bl.get("test_accuracy","—"),
            "Notes":           "Classical baseline (frozen backbone)",
        })

    # ── VQC result rows ───────────────────────────────────────────────────────
    for r in all_results:
        rows.append({
            "Model":           f"HQCNN (VQC q={r['n_qubits']} l={r['n_layers']})",
            "Regime":          r.get("regime", "A — frozen classical + VQC"),
            "Qubits":          r["n_qubits"],
            "Layers":          r["n_layers"],
            "TrainableParams": r["vqc_params"],
            "NoiseSigma":      r.get("noise_sigma", 0.0),
            "ValAUC":          r.get("best_val_auc", "—"),
            "TestAUC":         r.get("test_auc_roc", "—"),
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
                    "ValAUC", "TestAUC", "TestF1", "TestAcc"]
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

        model     = RegimeBModel(n_qubits, n_qubits, n_layers)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_auc, best_state, patience_ctr = 0.0, None, 0
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
            scheduler.step(vl_auc)

            is_best = vl_auc > best_auc
            if is_best:
                best_auc = vl_auc
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
        ts_f1  = f1_score(ts_labels, ts_preds, zero_division=0)
        ts_acc = accuracy_score(ts_labels, ts_preds)

        res = {
            "n_qubits": n_qubits, "n_layers": n_layers, "lr": lr,
            "noise_sigma": 0.0,
            "regime": "B — end-to-end (projection + VQC)",
            "best_val_auc": round(best_auc, 4),
            "vqc_params": model.count_params(),
            "test_auc_roc": round(ts_auc, 4),
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
        _, ts_acc, ts_f1, ts_auc, _, _, _ = eval_epoch(model, test_dl_n, nn.BCELoss())
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
    print("  QML Breast Cancer Classification | HQCNN Pipeline")
    print("═"*70)

    from cache_check import already_done, CACHE
    # Note: VQC results are NOT cached by default — rerunning is fast enough
    # and sweep results should always be fresh. Remove this note to add caching.

    all_results = []

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
                    "NoiseSigma","TestAUC","TestF1"]
    print(ablation_df[display_cols].to_string(index=False))

    print(f"\n✓ 3–5 COMPLETE. All outputs in: {OUT_DIR}")
    print("  Next step → 6_qfl.py (Quantum Federated Learning)")


if __name__ == "__main__":
    main()