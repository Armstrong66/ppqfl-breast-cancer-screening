"""
=============================================================================
6–7 — UNCERTAINTY QUANTIFICATION (UQ)
=============================================================================
Project : Quantum-Enhanced Hybrid Architectures for Mammographic Breast
          Cancer Classification in African and MENA Populations

Coverage:
  6 — MC-Dropout uncertainty on the classical baseline (MobileNetV2)
            Quantum measurement variance on trained HQCNN (VQC)
            Calibration analysis: reliability diagrams + ECE
  7 — UQ under noise: how does uncertainty grow as scanner quality drops?
            Regime A vs Regime B uncertainty comparison
            Final UQ summary plots for reporting

Motivation:
  In LMIC deployment contexts, knowing WHEN a model is uncertain is as
  clinically important as knowing its average accuracy. A model that is
  confidently wrong on degraded scanner images is more dangerous than one
  that flags its own uncertainty. This section quantifies that.

Two UQ sources are analysed in parallel:
  (1) Classical MC-Dropout:
        Run the MobileNetV2 head with Dropout kept ACTIVE at inference,
        repeat T=50 forward passes per sample, compute predictive mean
        and variance across passes. Standard Bayesian deep learning approach
        (Gal & Ghahramani, 2016).

  (2) Quantum Measurement Variance (shot-based):
        In a real quantum device, each circuit execution (shot) samples
        from the measurement distribution. Re-running the circuit N_SHOTS
        times per sample and measuring the variance of ⟨Z₀⟩ gives a
        natural quantum uncertainty estimate. We simulate this using
        PennyLane's "default.qubit" with finite shots.

Outputs (../ppqfl-breast-cancer-screening/outputs/uq_outputs/):
  mc_dropout_uncertainty.csv       ← per-sample UQ from classical model
  quantum_shot_variance.csv        ← per-sample UQ from VQC
  reliability_diagram_classical.png
  reliability_diagram_quantum.png
  uncertainty_vs_noise.png         ← UQ growth under scanner degradation
  regime_A_vs_B_uncertainty.png
  uq_summary.json                  ← ECE, mean uncertainty, calibration stats

Prerequisites: _3_5_vqc.py must have been run (Regime A checkpoint needed)
               _2_baseline.py checkpoint needed for MC-Dropout
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import json, pickle, warnings
from pathlib import Path
from copy import deepcopy
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

import pennylane as qml

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.calibration import calibration_curve

from pipeline_utils import seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
BASE         = PROJECT_ROOT / "outputs"
FEAT_DIR     = BASE / "feature_outputs"
VQC_CKPT_DIR = BASE / "vqc_outputs"
CNN_CKPT_DIR = BASE / "baseline_outputs"

BACKBONE     = "mobilenetv2"

def auto_detect_best_vqc_config(ckpt_dir: Path) -> tuple:
    """
    Scans regime_A checkpoint directory for the best VQC config by val AUC.
    Priority: best_run_manifest.json (written by multi-trial sweep) →
              history CSVs → first checkpoint found (fallback).
    Returns (n_qubits, n_layers, lr).
    """
    import re
    ckpt_dir = Path(ckpt_dir)
 
    # Priority 1: manifest written by multi-trial sweep
    manifest = ckpt_dir / "best_run_manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        cfg = (int(m["n_qubits"]), int(m["n_layers"]), float(m["lr"]))
        print(f"  Auto-detect: loaded from manifest → "
              f"q={cfg[0]} l={cfg[1]} lr={cfg[2]} "
              f"(val AUC={m.get('val_auc','?')})")
        return cfg
 
    # Priority 2: scan history CSVs for best val AUC
    best_auc   = -1.0
    best_config = None
    for ckpt_path in sorted(ckpt_dir.glob("vqc_q*.pt")):
        match = re.search(r"vqc_q(\d+)_l(\d+)_lr([\d.]+)\.pt", ckpt_path.name)
        if not match:
            continue
        nq, nl, lr = int(match.group(1)), int(match.group(2)), float(match.group(3))
        history_file = ckpt_dir / f"vqc_q{nq}_l{nl}_lr{lr}_history.csv"
        if history_file.exists():
            try:
                max_auc = pd.read_csv(history_file)["val_auc"].max()
                if max_auc > best_auc:
                    best_auc    = max_auc
                    best_config = (nq, nl, lr)
            except Exception:
                pass
        if best_config is None:          # fallback: first parseable checkpoint
            best_config = (nq, nl, lr)
            print(f"Fellback on {best_config}. Still couldn't find the right/actual configs from 3_5_vqc")
 
    if best_config is None:
        raise FileNotFoundError(
            f"No VQC checkpoints found in {ckpt_dir}. "
            "Run 3_5_vqc.py first."
        )
 
    print(f"  Auto-detect: best config from history → "
          f"q={best_config[0]} l={best_config[1]} lr={best_config[2]} "
          f"(val AUC={best_auc:.4f})")
    return best_config
 
# Dynamic Assignment
VQC_N_QUBITS, VQC_N_LAYERS, VQC_LR = auto_detect_best_vqc_config(VQC_CKPT_DIR / "regime_A")

OUT_DIR = BASE / "uq_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── UQ hyperparameters ────────────────────────────────────────────────────────
MC_T           = 50     # MC-Dropout forward passes per sample
N_SHOTS        = 100    # Quantum circuit shots per sample (shot-based variance)
N_BINS         = 10     # Reliability diagram bins
NOISE_SIGMAS   = [0.0, 0.05, 0.10, 0.20]
BATCH_SIZE     = 32
DEVICE         = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (identical logic to 3_5_vqc.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_test_split(n_qubits: int, noise_sigma: float = 0.0):
    """Load test split only (UQ is evaluated on the held-out test set)."""
    n       = n_qubits
    X_train = np.load(FEAT_DIR / f"features_train_pca{n}.npy")
    X_test  = np.load(FEAT_DIR / f"features_test_pca{n}.npy")
    y_test  = np.load(FEAT_DIR / "labels_test.npy")

    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)   # fit on train only
    X_test  = scaler.transform(X_test)

    if noise_sigma > 0:
        rng    = np.random.default_rng(42)
        X_test = np.clip(X_test + rng.normal(0, noise_sigma, X_test.shape), 0, 1)

    return X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VQC RECONSTRUCTION  (mirrors 3_5_vqc.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

def build_vqc_deterministic(n_qubits: int, n_layers: int):
    """Standard VQC with analytic expectation (no shots) — for point estimate."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(np.pi * inputs[i], wires=i)
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})


def build_vqc_shot_based(n_qubits: int, n_layers: int, n_shots: int):
    """
    Shot-based VQC: each call samples n_shots measurements and returns the
    empirical mean of ⟨Z₀⟩. Running this multiple times gives the shot variance.
    """
    dev = qml.device("default.qubit", wires=n_qubits, shots=n_shots)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(np.pi * float(inputs[i]), wires=i)
            for i in range(n_qubits):
                qml.RY(float(weights[layer, i]), wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})


class HQCNNClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers, shot_based=False, n_shots=100):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        if shot_based:
            self.vqc = build_vqc_shot_based(n_qubits, n_layers, n_shots)
        else:
            self.vqc = build_vqc_deterministic(n_qubits, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out  = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        prob = torch.sigmoid(out + self.bias)
        return prob


def load_vqc_checkpoint(n_qubits, n_layers, lr,
                        shot_based=False, n_shots=100) -> HQCNNClassifier:
    """Reconstruct HQCNNClassifier and load saved weights from regime_A."""
    model     = HQCNNClassifier(n_qubits, n_layers,
                                shot_based=shot_based, n_shots=n_shots)
    ckpt_name = f"vqc_q{n_qubits}_l{n_layers}_lr{lr}.pt"
    ckpt_path = VQC_CKPT_DIR / "regime_A" / ckpt_name   # ← regime_A subdir
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"VQC checkpoint not found: {ckpt_path}\n"
            "Run 3_5_vqc.py → run_regime_A() first."
        )
    state = torch.load(ckpt_path, map_location="cpu")
    # Load into deterministic model; shot-based shares same weights
    det_model = HQCNNClassifier(n_qubits, n_layers, shot_based=False)
    det_model.load_state_dict(state)
    # Copy weights to the target model
    model.bias.data = det_model.bias.data.clone()
    # VQC TorchLayer weights are stored as named parameters
    for (name, p_src), (_, p_dst) in zip(
        det_model.vqc.named_parameters(), model.vqc.named_parameters()
    ):
        p_dst.data = p_src.data.clone()
    print(f"  Loaded VQC checkpoint: {ckpt_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MC-DROPOUT FOR CLASSICAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_mobilenet_with_dropout(checkpoint_path: Path,
                                 dropout_p: float = 0.3) -> nn.Module:
    """
    Rebuild MobileNetV2 with Dropout explicitly in the head.
    The Dropout is already present from 2_baseline.py (p=0.3).
    We just need to ensure it stays ACTIVE at inference (model.train() mode
    for the head only, while BN layers stay in eval mode).
    """
    base = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    feat_dim = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(feat_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )
    if checkpoint_path.exists():
        base.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"  Loaded CNN checkpoint: {checkpoint_path}")
    else:
        print(f"  [WARNING] CNN checkpoint not found: {checkpoint_path}")
        print("  Using ImageNet weights — MC-Dropout results unreliable.")
    return base


def enable_mc_dropout(model: nn.Module):
    """
    Set model to eval() (freezes BatchNorm statistics) but keep
    Dropout layers active by switching them back to train() mode.
    This is the standard Gal & Ghahramani (2016) MC-Dropout protocol.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


def extract_cnn_features_for_uq(model: nn.Module) -> torch.Tensor:
    """
    Load the raw backbone features (1280-dim for MobileNetV2) for MC-Dropout.
    The backbone is frozen so its output is deterministic — only the head
    has Dropout, so uncertainty comes entirely from the stochastic head.
    Falls back to PCA features if raw features are unavailable.
    """
    head = model.classifier

    raw_path = FEAT_DIR / "features_test_raw.npy"
    if raw_path.exists():
        X_raw = np.load(raw_path)   # shape: (N, 1280)
        X_t   = torch.tensor(X_raw, dtype=torch.float32)
        print("  Using raw backbone features (1280-dim) for MC-Dropout head.")
    else:
        # Fallback: load PCA features (approximate — lower dim than head expects)
        print("  [INFO] Raw features not found; using PCA features as proxy.")
        print("         MC-Dropout uncertainty will be approximate.")
        X_pca = np.load(FEAT_DIR / f"features_test_pca{VQC_N_QUBITS}.npy")
        X_t   = torch.tensor(X_pca, dtype=torch.float32)
    return X_t, head


@torch.no_grad()
def mc_dropout_predict(head: nn.Module, X_t: torch.Tensor,
                       T: int = MC_T) -> tuple:
    """
    Run T stochastic forward passes through the classifier head.
    Returns:
      mean_probs  : (N,)  — predictive mean P(malignant)
      std_probs   : (N,)  — predictive std (epistemic uncertainty)
      all_probs   : (T, N) — all T predictions for downstream analysis
    """
    enable_mc_dropout(head)
    all_probs = []

    for _ in tqdm(range(T), desc="  MC-Dropout passes", leave=False):
        logits = head(X_t)                          # (N, 2)
        probs  = torch.softmax(logits, dim=1)[:, 1] # P(malignant)
        all_probs.append(probs.numpy())

    all_probs  = np.stack(all_probs)    # (T, N)
    mean_probs = all_probs.mean(axis=0) # (N,)
    std_probs  = all_probs.std(axis=0)  # (N,) — predictive uncertainty
    return mean_probs, std_probs, all_probs


# ══════════════════════════════════════════════════════════════════════════════
# 4.  QUANTUM MEASUREMENT VARIANCE
# ══════════════════════════════════════════════════════════════════════════════

def quantum_shot_variance(model_det: HQCNNClassifier,
                          X_test: np.ndarray,
                          n_repeats: int = 20,
                          n_shots: int = N_SHOTS) -> tuple:
    """
    Estimate quantum measurement uncertainty by running the circuit
    n_repeats times per sample with finite shots.

    For each sample x:
      - Run circuit n_repeats times, each with n_shots shots
      - Each run returns an empirical estimate of ⟨Z₀⟩
      - Variance across runs = quantum shot noise uncertainty

    Returns:
      mean_probs  : (N,) — mean P(malignant) across repeats
      var_probs   : (N,) — variance across repeats (quantum uncertainty)
    """
    print(f"  Computing quantum shot variance "
          f"(n_repeats={n_repeats}, n_shots={n_shots})...")
    print("  Note: this is slow — each sample runs the circuit "
          f"{n_repeats}× with {n_shots} shots each.")

    # Build shot-based VQC with same weights as deterministic model
    shot_model = HQCNNClassifier(
        model_det.n_qubits, model_det.n_layers,
        shot_based=True, n_shots=n_shots
    )
    shot_model.bias.data = model_det.bias.data.clone()
    for (_, p_src), (_, p_dst) in zip(
        model_det.vqc.named_parameters(),
        shot_model.vqc.named_parameters()
    ):
        p_dst.data = p_src.data.clone()
    shot_model.eval()

    N = len(X_test)
    all_runs = np.zeros((n_repeats, N))

    X_t = torch.tensor(X_test, dtype=torch.float32)

    for r in tqdm(range(n_repeats), desc="  Quantum repeats"):
        with torch.no_grad():
            run_probs = []
            for i in range(N):
                xi    = X_t[i].unsqueeze(0)
                prob  = shot_model(xi).item()
                run_probs.append(prob)
        all_runs[r] = run_probs

    mean_probs = all_runs.mean(axis=0)
    var_probs  = all_runs.var(axis=0)
    return mean_probs, var_probs, all_runs


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CALIBRATION: RELIABILITY DIAGRAM + ECE
# ══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = N_BINS) -> float:
    """
    ECE = Σ (|Bm| / N) × |acc(Bm) - conf(Bm)|
    Lower ECE = better calibrated model.
    A perfectly calibrated model has ECE = 0.
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    N        = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc    = y_true[mask].mean()
        conf   = y_prob[mask].mean()
        ece   += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray,
                              title: str, save_path: Path,
                              uncertainty: np.ndarray = None):
    """
    Reliability diagram (calibration curve) with optional uncertainty
    shading per confidence bin.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                              n_bins=N_BINS, strategy="uniform")
    ece = expected_calibration_error(y_true, y_prob)

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle(title, fontweight="bold", fontsize=12)

    # ── Reliability diagram ────────────────────────────────────────────────
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, marker="o", color="#4C72B0",
             lw=2, label=f"Model (ECE={ece:.4f})")
    ax1.fill_between(prob_pred, prob_true, prob_pred,
                     alpha=0.15, color="#DD8452", label="Calibration gap")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives (True)")
    ax1.set_title("Reliability Diagram")
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1])

    # ── Confidence histogram ───────────────────────────────────────────────
    ax2.hist(y_prob, bins=N_BINS, color="#4C72B0", edgecolor="white",
             alpha=0.8, density=True)
    if uncertainty is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.scatter(y_prob, uncertainty, alpha=0.3, s=8,
                         color="#DD8452", label="Uncertainty (std)")
        ax2_twin.set_ylabel("Predictive Std Dev", color="#DD8452")
        ax2_twin.tick_params(axis="y", colors="#DD8452")
    ax2.set_xlabel("Predicted P(malignant)")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return ece


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UNCERTAINTY UNDER NOISE (7)
# ══════════════════════════════════════════════════════════════════════════════

def uq_under_noise(vqc_model: HQCNNClassifier,
                   cnn_head: nn.Module,
                   X_raw_test: torch.Tensor,
                   y_test: np.ndarray) -> pd.DataFrame:
    """
    For each noise level σ, compute:
      - Mean predictive uncertainty (quantum shot variance)
      - Mean MC-Dropout uncertainty (classical)
      - AUC at that noise level

    This answers: "As scanner quality degrades, does uncertainty grow
    appropriately (well-calibrated) or does the model stay overconfident?"
    A well-behaved model should show rising uncertainty as noise rises.
    """
    print("\n  ── UQ Under Noise ──")
    records = []

    for sigma in NOISE_SIGMAS:
        X_test_n, _ = load_test_split(VQC_N_QUBITS, noise_sigma=sigma)

        # Quantum uncertainty at this noise level
        q_mean, q_var, _ = quantum_shot_variance(
            vqc_model, X_test_n, n_repeats=10, n_shots=50  # reduced for speed
        )
        q_auc = roc_auc_score(y_test, q_mean) if len(set(y_test)) > 1 else 0.0

        # Classical MC-Dropout uncertainty at this noise level
        # Inject noise into raw features too (if available), else use PCA proxy
        raw_path = FEAT_DIR / "features_test_raw.npy"
        if raw_path.exists():
            X_raw_n = np.load(raw_path).copy()
            if sigma > 0:
                rng = np.random.default_rng(42)
                # Scale noise relative to raw feature range
                feat_std  = X_raw_n.std()
                X_raw_n  += rng.normal(0, sigma * feat_std, X_raw_n.shape)
            X_cnn_t = torch.tensor(X_raw_n, dtype=torch.float32)
        else:
            X_cnn_t = X_raw_test  # fallback

        c_mean, c_std, _ = mc_dropout_predict(cnn_head, X_cnn_t, T=20)
        c_auc = roc_auc_score(y_test, c_mean) if len(set(y_test)) > 1 else 0.0

        records.append({
            "noise_sigma":         sigma,
            "quantum_mean_prob":   q_mean.mean(),
            "quantum_uncertainty": q_var.mean(),     # mean shot variance
            "quantum_auc":         round(q_auc, 4),
            "classical_mean_prob": c_mean.mean(),
            "classical_uncertainty": c_std.mean(),   # mean predictive std
            "classical_auc":       round(c_auc, 4),
        })
        print(f"    σ={sigma:.2f} | "
              f"Q-AUC={q_auc:.4f} Q-Unc={q_var.mean():.5f} | "
              f"C-AUC={c_auc:.4f} C-Unc={c_std.mean():.5f}")

    return pd.DataFrame(records)


def plot_uq_vs_noise(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Uncertainty Quantification Under Scanner Noise\n"
                 "Quantum (Shot Variance) vs Classical (MC-Dropout)",
                 fontweight="bold")

    # ── AUC degradation ───────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(df["noise_sigma"], df["quantum_auc"],
            marker="o", lw=2, color="#4C72B0", label="HQCNN (VQC)")
    ax.plot(df["noise_sigma"], df["classical_auc"],
            marker="s", lw=2, color="#DD8452", linestyle="--",
            label="Classical (MobileNetV2)")
    ax.set_xlabel("Gaussian Noise σ"); ax.set_ylabel("Test AUC-ROC")
    ax.set_title("AUC Degradation Under Noise")
    ax.legend(); ax.set_ylim([0.5, 1.02]); ax.grid(True, alpha=0.3)

    # ── Uncertainty growth ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(df["noise_sigma"], df["quantum_uncertainty"],
            marker="o", lw=2, color="#4C72B0", label="Quantum shot variance")
    ax.plot(df["noise_sigma"], df["classical_uncertainty"],
            marker="s", lw=2, color="#DD8452", linestyle="--",
            label="MC-Dropout std")
    ax.set_xlabel("Gaussian Noise σ")
    ax.set_ylabel("Mean Predictive Uncertainty")
    ax.set_title("Uncertainty Growth Under Noise\n"
                 "(Well-calibrated model: uncertainty ↑ as noise ↑)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REGIME A vs REGIME B UNCERTAINTY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_regimes_uncertainty(y_test: np.ndarray):
    """
    Load Regime A and Regime B checkpoints, compute shot variance for both,
    and plot side-by-side. Tests whether end-to-end training produces a
    more or less uncertain model than frozen-classical + VQC.
    """
    print("\n  ── Regime A vs B Uncertainty ──")
    X_test, _ = load_test_split(VQC_N_QUBITS, noise_sigma=0.0)
    results   = {}

    for regime, ckpt_dir, label in [
        ("A", VQC_CKPT_DIR / "regime_A", "Frozen Classical + VQC"),
        ("B", VQC_CKPT_DIR / "regime_B", "End-to-End (Proj + VQC)"),
    ]:
        ckpt = ckpt_dir / f"vqc_q{VQC_N_QUBITS}_l{VQC_N_LAYERS}_lr{VQC_LR}.pt"
        if not ckpt.exists():
            # Regime B uses different lr
            alts = list(ckpt_dir.glob(f"vqc_q{VQC_N_QUBITS}_l{VQC_N_LAYERS}_*.pt"))
            if alts:
                ckpt = alts[0]
            else:
                print(f"  [SKIP] No checkpoint for Regime {regime}: {ckpt_dir}")
                continue

        model = HQCNNClassifier(VQC_N_QUBITS, VQC_N_LAYERS, shot_based=False)
        state = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Regime B uses RegimeBModel with a projection layer —
            # extract only the VQC part
            vqc_state = {k.replace("hqcnn.", ""): v
                         for k, v in state.items() if "hqcnn" in k}
            model.load_state_dict(vqc_state, strict=False)

        mean_p, var_p, _ = quantum_shot_variance(
            model, X_test, n_repeats=10, n_shots=50
        )
        auc = roc_auc_score(y_test, mean_p) if len(set(y_test)) > 1 else 0.0
        results[regime] = {
            "label": label, "mean_prob": mean_p,
            "var": var_p, "auc": auc
        }
        print(f"    Regime {regime}: AUC={auc:.4f}  "
              f"Mean uncertainty={var_p.mean():.5f}")

    if len(results) < 2:
        print("  Skipping regime comparison plot (need both A and B checkpoints).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regime A vs B — Quantum Uncertainty Distribution",
                 fontweight="bold")
    colors = {"A": "#4C72B0", "B": "#DD8452"}

    for ax, (regime, data) in zip(axes, results.items()):
        ax.hist(data["var"], bins=20, color=colors[regime],
                edgecolor="white", alpha=0.8, density=True)
        ax.axvline(data["var"].mean(), color="red", linestyle="--",
                   label=f"Mean={data['var'].mean():.5f}")
        ax.set_title(f"Regime {regime}: {data['label']}\nAUC={data['auc']:.4f}")
        ax.set_xlabel("Shot Variance (Quantum Uncertainty per Sample)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = OUT_DIR / "regime_A_vs_B_uncertainty.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TEMPERATURE SCALING — post-hoc VQC calibration
# ══════════════════════════════════════════════════════════════════════════════

def temperature_scale_vqc(vqc_model: HQCNNClassifier,
                           X_val: np.ndarray,
                           y_val: np.ndarray) -> tuple:
    """
    Learn a scalar temperature T on the validation set that minimises
    negative log-likelihood of the re-scaled VQC outputs.

    Mechanism:
      Raw VQC output: p = sigmoid(logit + bias)
      Scaled output:  p_T = sigmoid((logit + bias) / T)

    Interpretation:
      - T > 1  => logits are divided by T > 1, producing smaller logits and
                softer (less confident) probabilities — useful to correct
                overconfident models.
      - T = 1  => no change.
      - T < 1  => logits are amplified (division by a number < 1), producing
                sharper (more confident) probabilities.

    T is fitted on the validation set (by minimising NLL) and then applied
    to the test set to produce calibrated probabilities.

    Returns:
      T_opt       : float — optimal temperature
      probs_scaled: ndarray — re-calibrated test probabilities
    """
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # Collect raw logits from VQC (before sigmoid) on val set
    vqc_model.eval()
    with torch.no_grad():
        raw_logits = []
        for i in range(len(X_val_t)):
            xi   = X_val_t[i]
            z    = vqc_model.vqc(xi)          # ⟨Z₀⟩ ∈ [-1, 1]
            raw_logits.append((z + vqc_model.bias).item())
    logits_val = torch.tensor(raw_logits, dtype=torch.float32)
    labels_val = torch.tensor(y_val,      dtype=torch.float32)

    # Optimise T via NLL on val set
    T = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=500)

    def nll_closure():
        optimizer.zero_grad()
        scaled = torch.sigmoid(logits_val / T.clamp(min=0.05))
        loss   = torch.nn.BCELoss()(scaled, labels_val)
        loss.backward()
        return loss

    optimizer.step(nll_closure)
    # Keep the final temperature within the optimisation bounds used during fitting.
    T_opt = float(T.clamp(min=0.05).item())

    # Re-fit scaler on train to avoid leakage
    X_train_pca = np.load(FEAT_DIR / f"features_train_pca{VQC_N_QUBITS}.npy")
    scaler      = MinMaxScaler().fit(X_train_pca)
    X_test_raw  = np.load(FEAT_DIR / f"features_test_pca{VQC_N_QUBITS}.npy")
    X_test_sc   = scaler.transform(X_test_raw)
    X_test_t    = torch.tensor(X_test_sc, dtype=torch.float32)

    # Generate test predictions using the optimized temperature
    with torch.no_grad():
        raw_logits_test = []
        for i in range(len(X_test_t)):
            xi = X_test_t[i]
            z  = vqc_model.vqc(xi)
            raw_logits_test.append((z + vqc_model.bias).item())
    logits_test = torch.tensor(raw_logits_test)
    logits_test_scaled = logits_test / T_opt
    probs_scaled = torch.sigmoid(logits_test_scaled).numpy()

    return T_opt, probs_scaled, logits_test_scaled


def plot_calibration_comparison(y_true, probs_before, probs_after,
                                 T_opt, save_path: Path):
    """Reliability diagram: before vs after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"VQC Temperature Scaling (T={T_opt:.3f})\n"
                 "Reliability Diagram Before vs After Calibration",
                 fontweight="bold")

    for ax, probs, title in [
        (axes[0], probs_before, "Before scaling"),
        (axes[1], probs_after,  f"After scaling (T={T_opt:.3f})"),
    ]:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(y_true, probs,
                                                 n_bins=N_BINS,
                                                 strategy="uniform")
        ece = expected_calibration_error(y_true, probs)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(mean_pred, frac_pos, marker="o", color="#4C72B0",
                lw=2, label=f"ECE={ece:.4f}")
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.15, color="#DD8452", label="Gap")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print("  6–7 — UNCERTAINTY QUANTIFICATION PIPELINE")
    print("  QML Breast Cancer Classification | MC-Dropout + Quantum Variance")
    print("═"*70)

    y_test = np.load(FEAT_DIR / "labels_test.npy")
    summary = {}

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading models...")
    seed_everything(42)

    # VQC (deterministic, best Regime A checkpoint)
    vqc_model = load_vqc_checkpoint(VQC_N_QUBITS, VQC_N_LAYERS, VQC_LR,
                                     shot_based=False)
    vqc_model.eval()

    # Classical head for MC-Dropout
    cnn_ckpt   = CNN_CKPT_DIR / f"{BACKBONE}_best.pt"
    cnn_model  = build_mobilenet_with_dropout(cnn_ckpt)
    X_raw_t, cnn_head = extract_cnn_features_for_uq(cnn_model)

    # ── Point-estimate metrics (sanity check) ─────────────────────────────
    print("\n[2/7] Point-estimate sanity check on test set...")
    X_test_q, _ = load_test_split(VQC_N_QUBITS)
    X_t_q = torch.tensor(X_test_q, dtype=torch.float32)
    with torch.no_grad():
        vqc_probs = vqc_model(X_t_q).numpy()
    vqc_auc = roc_auc_score(y_test, vqc_probs)
    print(f"  VQC point-estimate AUC: {vqc_auc:.4f}")

    with torch.no_grad():
        cnn_model.eval()
        cnn_logits = cnn_head(X_raw_t)
        cnn_probs  = torch.softmax(cnn_logits, dim=1)[:, 1].numpy()
    cnn_auc = roc_auc_score(y_test, cnn_probs)
    print(f"  CNN point-estimate AUC: {cnn_auc:.4f}")

    # ── MC-Dropout (classical) ────────────────────────────────────────────
    print(f"\n[3/7] MC-Dropout ({MC_T} passes)...")
    mc_mean, mc_std, mc_all = mc_dropout_predict(cnn_head, X_raw_t, T=MC_T)
    mc_auc  = roc_auc_score(y_test, mc_mean)
    mc_ece  = plot_reliability_diagram(
        y_test, mc_mean,
        title=f"Classical MobileNetV2 — MC-Dropout (T={MC_T})\nAUC={mc_auc:.4f}",
        save_path=OUT_DIR / "reliability_diagram_classical.png",
        uncertainty=mc_std
    )
    pd.DataFrame({
        "y_true": y_test, "mc_mean_prob": mc_mean, "mc_std": mc_std
    }).to_csv(OUT_DIR / "mc_dropout_uncertainty.csv", index=False)
    print(f"  MC-Dropout — AUC: {mc_auc:.4f}  ECE: {mc_ece:.4f}  "
          f"Mean uncertainty: {mc_std.mean():.4f}")
    summary["classical_mc_dropout"] = {
        "auc": round(mc_auc, 4), "ece": round(mc_ece, 4),
        "mean_uncertainty": round(float(mc_std.mean()), 6),
        "T": MC_T
    }

    # ── Quantum shot variance ─────────────────────────────────────────────
    print(f"\n[4/7] Quantum shot variance (n_shots={N_SHOTS})...")
    q_mean, q_var, q_all = quantum_shot_variance(
        vqc_model, X_test_q, n_repeats=20, n_shots=N_SHOTS
    )
    q_auc = roc_auc_score(y_test, q_mean)
    q_ece = plot_reliability_diagram(
        y_test, q_mean,
        title=(f"HQCNN VQC q={VQC_N_QUBITS} l={VQC_N_LAYERS} — "
               f"Shot Variance (shots={N_SHOTS})\nAUC={q_auc:.4f}"),
        save_path=OUT_DIR / "reliability_diagram_quantum.png",
        uncertainty=np.sqrt(q_var)
    )
    pd.DataFrame({
        "y_true": y_test, "q_mean_prob": q_mean, "q_var": q_var
    }).to_csv(OUT_DIR / "quantum_shot_variance.csv", index=False)
    print(f"  Quantum — AUC: {q_auc:.4f}  ECE: {q_ece:.4f}  "
          f"Mean shot var: {q_var.mean():.6f}")
    summary["quantum_shot_variance"] = {
        "auc": round(q_auc, 4), "ece": round(q_ece, 4),
        "mean_shot_variance": round(float(q_var.mean()), 8),
        "n_shots": N_SHOTS, "n_repeats": 20
    }

    # ── UQ under noise ────────────────────────────────────────────────────
    print("\n[5/7] UQ under scanner noise...")
    noise_df = uq_under_noise(vqc_model, cnn_head, X_raw_t, y_test)
    noise_df.to_csv(OUT_DIR / "uq_under_noise.csv", index=False)
    plot_uq_vs_noise(noise_df, OUT_DIR / "uncertainty_vs_noise.png")
    summary["uq_under_noise"] = noise_df.to_dict(orient="records")

    # ── Regime A vs B ─────────────────────────────────────────────────────
    print("\n[6/7] Regime A vs B uncertainty comparison...")
    compare_regimes_uncertainty(y_test)

    # ── Temperature scaling ───────────────────────────────────────────────
    print(f"\n[7/7] Temperature scaling (post-hoc VQC calibration)...")
    print("  Fitting temperature T on validation set to reduce overconfidence...")
    X_val_pca    = np.load(FEAT_DIR / f"features_val_pca{VQC_N_QUBITS}.npy")
    X_train_pca  = np.load(FEAT_DIR / f"features_train_pca{VQC_N_QUBITS}.npy")
    y_val        = np.load(FEAT_DIR / "labels_val.npy")
    scaler_fit   = MinMaxScaler().fit(X_train_pca)
    X_val_scaled = scaler_fit.transform(X_val_pca)

    T_opt, probs_scaled, logits_scaled = temperature_scale_vqc(
        vqc_model, X_val_scaled, y_val)
    ece_after  = expected_calibration_error(y_test, probs_scaled)
    auc_scaled = roc_auc_score(y_test, logits_scaled)
    print(f"  Optimal T     : {T_opt:.4f}  "
          f"({'overconfident → softened' if T_opt > 1 else 'underconfident → sharpened'})")
    print(f"  ECE before    : {q_ece:.4f}")
    print(f"  ECE after     : {ece_after:.4f}  "
          f"({'improved ✓' if ece_after < q_ece else 'no improvement'})")
    print(f"  AUC (scaled)  : {auc_scaled:.4f}  (should match pre-scaling: {q_auc:.4f})")

    plot_calibration_comparison(
        y_test, vqc_probs, probs_scaled, T_opt,
        OUT_DIR / "temperature_scaling_calibration.png"
    )
    pd.DataFrame({
        "y_true": y_test,
        "q_prob_raw": vqc_probs,
        "q_prob_scaled": probs_scaled,
    }).to_csv(OUT_DIR / "temperature_scaled_probs.csv", index=False)

    summary["temperature_scaling"] = {
        "T_optimal":  round(T_opt, 4),
        "ece_before": round(q_ece,     4),
        "ece_after":  round(ece_after, 4),
        "auc_scaled": round(auc_scaled, 4),
        "note": ("T > 1 means VQC was overconfident; "
                 "temperature scaling softens sigmoid outputs. "
                 "AUC is invariant to monotone rescaling — "
                 "confirm auc_scaled ≈ quantum_shot_variance auc.")
    }

    # ── Save summary ──────────────────────────────────────────────────────
    with open(OUT_DIR / "uq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═"*70)
    print("  UNCERTAINTY QUANTIFICATION COMPLETE")
    print(f"  Outputs saved to: {OUT_DIR}")
    print("\n  Key metrics:")
    print(f"    Classical ECE      : {mc_ece:.4f} (lower = better calibrated)")
    print(f"    Quantum ECE raw    : {q_ece:.4f}")
    print(f"    Quantum ECE scaled : {ece_after:.4f}  (T={T_opt:.3f})")
    print(f"    Classical mean unc : {mc_std.mean():.4f} (MC-Dropout std)")
    print(f"    Quantum mean unc   : {q_var.mean():.6f} (shot variance)")
    print("\n  Interpretation guide:")
    print("    ECE < 0.05  → well calibrated")
    print("    ECE 0.05–0.15 → moderate miscalibration (common in small datasets)")
    print("    ECE > 0.15  → overconfident — temperature scaling applied above")
    print("\n  Next step → _8_9_qfl.py (Quantum Federated Learning)")


if __name__ == "__main__":
    main()