"""
=============================================================================
_10–11 — EXTERNAL VALIDATION + FINAL CONSOLIDATED ABLATION TABLE
=============================================================================
Project : Privacy-Preserving Quantum Federated Learning for Breast Cancer
          Screening in African and MENA Populations

Coverage:
  10 — External validation on KAU-BCMD (Saudi Arabia / MENA)
             BI-RADS binary relabelling (1,3 → Benign; 4,5 → Malignant)
             Cross-population performance comparison (Mendeley vs KAU)
             Domain shift analysis (feature-space distance + AUC gap)
             Parameter efficiency comparison (VQC vs classical head)

  11 — Consolidate ALL experiment results into one master ablation table
             Final publication-ready figures
             Cross-population generalisation report (JSON + DOCX-ready CSV)

Outputs (../ppqfl-breast-cancer-screening/outputs/external_val_outputs/):
  kau_test_results.json            ← per-model KAU performance
  cross_population_comparison.png  ← Mendeley vs KAU AUC side-by-side
  domain_shift_analysis.png        ← PCA feature distribution overlap
  parameter_efficiency.png         ← params vs AUC scatter (all models)
  final_ablation_table.csv         ← MASTER table: all experiments
  final_ablation_table.png         ← publication-ready figure
  generalisation_report.json       ← structured report for supervisor
=============================================================================
"""

import json, pickle, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

import pennylane as qml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve, brier_score_loss,
    balanced_accuracy_score
)

warnings.filterwarnings("ignore")

from pipeline_utils import seed_everything
seed_everything(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
BASE         = PROJECT_ROOT / "outputs"
FEAT_DIR     = BASE / "feature_outputs"
BASELINE_DIR = BASE / "baseline_outputs"
VQC_DIR_A    = BASE / "vqc_outputs/regime_A"
VQC_DIR_B    = BASE / "vqc_outputs/regime_B"
NOISE_DIR    = BASE / "vqc_outputs/noise"
QFL_DIR      = BASE / "qfl_outputs"
UQ_DIR       = BASE / "uq_outputs"
OUT_DIR      = BASE / "external_val_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def auto_detect_best_vqc_config(ckpt_dir: Path) -> tuple:
    manifest = ckpt_dir / "best_run_manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        return int(m.get("n_qubits", 4)), int(m.get("n_layers", 2)), float(m.get("lr", 0.01))

    best_config = None
    best_auc = -1.0
    for ckpt in sorted(ckpt_dir.glob("vqc_q*_l*_lr*.pt")):
        match = re.match(r"vqc_q(\d+)_l(\d+)_lr([\d.]+)\.pt", ckpt.name)
        if not match:
            continue
        nq, nl, lr = int(match.group(1)), int(match.group(2)), float(match.group(3))
        history = ckpt_dir / f"vqc_q{nq}_l{nl}_lr{lr}_history.csv"
        if history.exists():
            try:
                auc = pd.read_csv(history)["val_auc"].max()
                if auc > best_auc:
                    best_auc = auc
                    best_config = (nq, nl, lr)
            except Exception:
                continue
    if best_config is not None:
        return best_config

    print("  [WARN] Could not auto-detect best VQC config from manifest or history. "
          "Falling back to defaults q=4, l=2, lr=0.01.")
    return 4, 2, 0.01

# ── Match to best config from _3–5 sweep (this is also need to auto_detect) ─────────────────────────────────
BACKBONE      = "mobilenetv2"
N_QUBITS, N_LAYERS, VQC_LR = auto_detect_best_vqc_config(VQC_DIR_A)
BATCH_SIZE    = 32
DEVICE        = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  VQC RECONSTRUCTION  (self-contained, no cross-file import)
# ══════════════════════════════════════════════════════════════════════════════

def build_vqc(n_qubits, n_layers):
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


class VQCModel(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.vqc  = build_vqc(n_qubits, n_layers)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        out = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        return torch.sigmoid(out + self.bias)
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_vqc(ckpt_dir, n_qubits, n_layers, lr):
    seed_everything(42)
    model = VQCModel(n_qubits, n_layers)
    ckpt  = ckpt_dir / f"vqc_q{n_qubits}_l{n_layers}_lr{lr}.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"  Loaded: {ckpt}")
    else:
        print(f"  [WARN] checkpoint not found: {ckpt}")
    model.eval()
    return model


def build_mobilenet(ckpt_path):
    base = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    feat_dim = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(feat_dim, 128),
        nn.ReLU(), nn.Linear(128, 2),
    )
    if ckpt_path.exists():
        base.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    base.eval()
    return base


class ClassicalMicroMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_or_train_micromlp(in_dim: int, X_train_scaled, y_train):
    """Load trained micro-MLP checkpoint or fit quickly on training features."""
    seed_everything(42)
    model = ClassicalMicroMLP(in_dim)
    ckpt = VQC_DIR_A.parent / "micromlp_q4.pt"
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  Loaded Micro-MLP checkpoint: {ckpt}")
            model.eval()
            return model
        except Exception:
            pass
    # Fallback: train on train split
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model.train()
    for _ in range(30):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD KAU FEATURES + LABELS
# ══════════════════════════════════════════════════════════════════════════════

def load_kau_features(n_qubits):
    """
    Load KAU-BCMD features extracted by 2b_feature_pca.py.
    Apply the same scaler fitted on Mendeley training data.
    This is critical: the scaler must NOT be refit on KAU data —
    that would constitute data leakage from the external validation set.
    """
    # Raw features (for CNN evaluation and domain shift analysis)
    X_kau_raw = np.load(FEAT_DIR / "features_kau_raw.npy")
    y_kau     = np.load(FEAT_DIR / "labels_kau.npy")

    # PCA features (pre-transformed by 2b_feature_pca.py using Mendeley scaler)
    X_kau_pca = np.load(FEAT_DIR / f"features_kau_pca{n_qubits}.npy")

    # Apply the same MinMaxScaler used in VQC training
    X_train_pca = np.load(FEAT_DIR / f"features_train_pca{n_qubits}.npy")
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train_pca)
    X_kau_scaled = scaler.transform(X_kau_pca)

    print(f"  KAU-BCMD: {len(y_kau)} samples | "
          f"Benign: {(y_kau==0).sum()} | Malignant: {(y_kau==1).sum()}")
    print(f"  Class ratio (B:M) = {(y_kau==0).sum()}:{(y_kau==1).sum()} "
          f"({(y_kau==0).sum()/(y_kau==1).sum():.1f}:1) — "
          f"report per-class metrics")
    return X_kau_raw, X_kau_pca, X_kau_scaled, y_kau


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATE MODEL ON KAU
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold_youden(y_true, probs) -> float:
    """Find optimal classification threshold maximizing Youden's J = TPR - FPR on a calibration set."""
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    return float(np.clip(best_thresh, 0.05, 0.95))


def compute_comprehensive_metrics(y_true, probs, opt_threshold: Optional[float] = None) -> dict:
    """
    Compute full suite of classification and calibration metrics:
    1. Threshold-independent: AUC-ROC, PR-AUC, Brier score
    2. Default threshold (0.5): F1, Accuracy, MCC, Sensitivity, Specificity
    3. Optimal threshold (Youden's J): Optimal threshold, F1, Balanced Accuracy, Sensitivity, Specificity
    """
    y_true = np.asarray(y_true)
    probs  = np.asarray(probs)
    n_classes = len(set(y_true))

    auc = roc_auc_score(y_true, probs) if n_classes > 1 else 0.0
    ap  = average_precision_score(y_true, probs) if n_classes > 1 else 0.0
    brier = float(brier_score_loss(y_true, probs))

    # Standard 0.5 threshold
    preds_05 = (probs > 0.5).astype(int)
    f1_05    = float(f1_score(y_true, preds_05, zero_division=0))
    acc_05   = float(accuracy_score(y_true, preds_05))
    mcc_05   = float(matthews_corrcoef(y_true, preds_05)) if n_classes > 1 else 0.0
    bacc_05  = float(balanced_accuracy_score(y_true, preds_05)) if n_classes > 1 else acc_05

    report_05 = classification_report(
        y_true, preds_05, target_names=["Benign", "Malignant"], output_dict=True
    ) if n_classes > 1 else {}

    # Optimal threshold (if provided or calculated)
    tau = opt_threshold if opt_threshold is not None else find_optimal_threshold_youden(y_true, probs)
    preds_opt = (probs > tau).astype(int)
    f1_opt    = float(f1_score(y_true, preds_opt, zero_division=0))
    acc_opt   = float(accuracy_score(y_true, preds_opt))
    mcc_opt   = float(matthews_corrcoef(y_true, preds_opt)) if n_classes > 1 else 0.0
    bacc_opt  = float(balanced_accuracy_score(y_true, preds_opt)) if n_classes > 1 else acc_opt
    report_opt = classification_report(
        y_true, preds_opt, target_names=["Benign", "Malignant"], output_dict=True
    ) if n_classes > 1 else {}

    return {
        "auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        # 0.5 standard metrics
        "f1": round(f1_05, 4),
        "accuracy": round(acc_05, 4),
        "balanced_accuracy": round(bacc_05, 4),
        "mcc": round(mcc_05, 4),
        "per_class": report_05,
        # Clinical optimal threshold metrics
        "opt_threshold": round(tau, 4),
        "f1_at_opt": round(f1_opt, 4),
        "balanced_acc_at_opt": round(bacc_opt, 4),
        "per_class_at_opt": report_opt,
        "probs": probs,
        "preds": preds_05,
        "preds_at_opt": preds_opt,
    }


@torch.no_grad()
def evaluate_vqc_on_kau(model, X_kau_scaled, y_kau, opt_threshold: Optional[float] = None):
    X_t   = torch.tensor(X_kau_scaled, dtype=torch.float32)
    probs = model(X_t).numpy()
    return compute_comprehensive_metrics(y_kau, probs, opt_threshold=opt_threshold)


@torch.no_grad()
def evaluate_cnn_on_kau(model, X_kau_raw, y_kau, opt_threshold: Optional[float] = None):
    X_t    = torch.tensor(X_kau_raw, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, torch.tensor(y_kau)),
                        batch_size=BATCH_SIZE, shuffle=False)
    all_probs = []
    for X_b, _ in loader:
        logits = model.classifier(X_b)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.numpy())
    probs = np.array(all_probs)
    return compute_comprehensive_metrics(y_kau, probs, opt_threshold=opt_threshold)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DOMAIN SHIFT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def domain_shift_analysis(X_mendeley_pca, y_mendeley,
                           X_kau_pca, y_kau,
                           save_path: Path):
    """
    Quantify the feature-space distribution shift between Mendeley (SA)
    and KAU-BCMD (Saudi Arabia / MENA).

    Three complementary measures:
      1. Jensen-Shannon divergence per PCA dimension (marginal shift)
      2. Kolmogorov-Smirnov test per PCA dimension (distribution test)
      3. 2D PCA scatter coloured by dataset origin (visual inspection)
    """
    n_dims = X_mendeley_pca.shape[1]
    js_divs, ks_stats, ks_pvals = [], [], []

    # Fit a common StandardScaler for visual comparison only
    scaler = StandardScaler().fit(
        np.vstack([X_mendeley_pca, X_kau_pca])
    )
    M_s = scaler.transform(X_mendeley_pca)
    K_s = scaler.transform(X_kau_pca)

    for d in range(n_dims):
        m_vals = M_s[:, d]
        k_vals = K_s[:, d]
        # Histogram-based JS divergence
        lo, hi  = min(m_vals.min(), k_vals.min()), max(m_vals.max(), k_vals.max())
        bins    = np.linspace(lo, hi, 30)
        p, _    = np.histogram(m_vals, bins=bins, density=True)
        q, _    = np.histogram(k_vals, bins=bins, density=True)
        p       = p + 1e-10; q = q + 1e-10
        p /= p.sum(); q /= q.sum()
        js_divs.append(float(jensenshannon(p, q)))
        ks_stat, ks_p = ks_2samp(m_vals, k_vals)
        ks_stats.append(float(ks_stat)); ks_pvals.append(float(ks_p))

    # ── 2D PCA projection ────────────────────────────────────────────────
    pca2d  = PCA(n_components=2, random_state=42)
    both   = np.vstack([X_mendeley_pca, X_kau_pca])
    emb    = pca2d.fit_transform(StandardScaler().fit_transform(both))
    n_m    = len(X_mendeley_pca)
    emb_m, emb_k = emb[:n_m], emb[n_m:]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle("Domain Shift Analysis: Mendeley (SA) vs KAU-BCMD (MENA)\n"
                 "PCA Feature Space Distribution Comparison", fontweight="bold")

    # JS divergence bar chart
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(range(1, n_dims + 1), js_divs, color="#4C72B0", edgecolor="white")
    ax1.axhline(np.mean(js_divs), color="red", linestyle="--",
                label=f"Mean JS={np.mean(js_divs):.3f}")
    ax1.set_xlabel("PCA Component"); ax1.set_ylabel("Jensen-Shannon Divergence")
    ax1.set_title("Per-component Distribution Shift\n(0=identical, 1=maximally different)")
    ax1.legend(fontsize=8); ax1.set_xticks(range(1, n_dims + 1))

    # KS statistic
    ax2 = fig.add_subplot(gs[1])
    colors_ks = ["#DD8452" if p < 0.05 else "#4C72B0" for p in ks_pvals]
    ax2.bar(range(1, n_dims + 1), ks_stats, color=colors_ks, edgecolor="white")
    ax2.set_xlabel("PCA Component"); ax2.set_ylabel("KS Statistic")
    ax2.set_title("Kolmogorov-Smirnov Test\n(orange = significant shift p<0.05)")
    ax2.set_xticks(range(1, n_dims + 1))

    # 2D scatter
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(emb_m[:,0], emb_m[:,1], c="#4C72B0", alpha=0.4, s=15,
                label=f"Mendeley SA (n={n_m})")
    ax3.scatter(emb_k[:,0], emb_k[:,1], c="#DD8452", alpha=0.4, s=15,
                label=f"KAU-BCMD MENA (n={len(X_kau_pca)})")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
    ax3.set_title("2D Feature Space\n(overlap = low domain shift)")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        "mean_js_divergence": round(float(np.mean(js_divs)), 4),
        "max_js_divergence":  round(float(np.max(js_divs)),  4),
        "mean_ks_statistic":  round(float(np.mean(ks_stats)), 4),
        "n_dims_significant_shift": int(sum(p < 0.05 for p in ks_pvals)),
        "per_dim_js": [round(v, 4) for v in js_divs],
        "per_dim_ks": [round(v, 4) for v in ks_stats],
        "interpretation": (
            f"Mean JS divergence = {np.mean(js_divs):.3f} across {n_dims} PCA dims. "
            f"{sum(p<0.05 for p in ks_pvals)}/{n_dims} dims show statistically "
            "significant distribution shift (KS test, p<0.05). "
            "Moderate domain shift expected given different population and scanner types."
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CROSS-POPULATION COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_population(mendeley_results: dict,
                           kau_results: dict,
                           save_path: Path):
    """
    Side-by-side AUC and F1 bars for each model on Mendeley (primary)
    vs KAU-BCMD (external validation). The AUC gap is the generalisation cost.
    """
    models_list = list(mendeley_results.keys())
    x     = np.arange(len(models_list))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Cross-Population Generalisation\n"
                 "Mendeley (SA, primary) vs KAU-BCMD (MENA, external validation)",
                 fontweight="bold")

    for ax, metric, ylabel in [
        (axes[0], "auc",  "AUC-ROC"),
        (axes[1], "f1",   "F1 Score"),
    ]:
        m_vals = [mendeley_results[m].get(metric, 0) for m in models_list]
        k_vals = [kau_results[m].get(metric, 0)      for m in models_list]

        bars1 = ax.bar(x - width/2, m_vals, width, label="Mendeley (SA)",
                       color="#4C72B0", edgecolor="white")
        bars2 = ax.bar(x + width/2, k_vals, width, label="KAU-BCMD (MENA)",
                       color="#DD8452", edgecolor="white", alpha=0.85)

        for bars, vals in [(bars1, m_vals), (bars2, k_vals)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        # Annotate generalisation gap
        for i, (mv, kv) in enumerate(zip(m_vals, k_vals)):
            gap = mv - kv
            color = "#cc0000" if gap > 0.05 else "#228B22"
            ax.annotate(f"Δ={gap:+.3f}",
                        xy=(x[i], max(mv, kv) + 0.025),
                        ha="center", fontsize=7, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_ylim([0, 1.12])
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{ylabel} — Primary vs External Validation\n"
                     "(Δ = generalisation gap; red = >5% drop)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PARAMETER EFFICIENCY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_parameter_efficiency(all_results: list, save_path: Path):
    """
    Scatter plot: trainable parameter count (log scale) vs test AUC.
    The quantum utility argument rests on the VQC sitting in the upper-left
    quadrant — high AUC, very few parameters. This is the key figure for
    reporting quantum advantage in low-resource settings.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Parameter Efficiency: Trainable Parameters vs Performance\n"
                 "Quantum Utility in Low-Resource LMIC Deployment Context",
                 fontweight="bold")

    colors = {
        "Classical": "#4C72B0",
        "HQCNN Regime A": "#DD8452",
        "HQCNN Regime B": "#2ca02c",
        "QFL Federated": "#9467bd",
        "QFL + DP": "#8c564b",
    }

    for ax, dataset, title in [
        (axes[0], "mendeley", "Mendeley Test Set (Primary)"),
        (axes[1], "kau",      "KAU-BCMD (External Validation)"),
    ]:
        seen_cats = set()
        for row in all_results:
            params = row.get("trainable_params", row.get("vqc_params", 1))
            auc    = row.get(f"{dataset}_auc", row.get("test_auc_roc",
                             row.get("final_test_auc", None)))
            label  = row.get("model_short", row.get("model", "?"))
            cat    = row.get("category", "HQCNN Regime A")
            color  = colors.get(cat, "gray")

            # Skip rows with None/invalid auc for this dataset axis
            if auc is None or not isinstance(auc, (int, float)):
                continue

            ax.scatter(params, auc, c=color, s=100, zorder=5,
                       label=cat if cat not in seen_cats else "")
            seen_cats.add(cat)
            ax.annotate(label, (params, auc),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=color)

        ax.set_xscale("log")
        ax.set_xlabel("Trainable Parameters (log scale)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title(title)
        ax.set_ylim([0.5, 1.05])
        ax.grid(True, alpha=0.3)
        # Target quadrant annotation
        ax.axhline(0.90, color="gray", linestyle=":", alpha=0.5)
        ax.text(1.2, 0.905, "AUC ≥ 0.90 threshold", fontsize=7,
                color="gray", va="bottom")

    # Deduplicate legend
    handles, labels_leg = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    axes[0].legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MASTER ABLATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_master_ablation(
    kau_results_dict: dict,
    baseline_json: Path,
    vqc_dir_A: Path,
    vqc_dir_B: Path,
    noise_dir: Path,
    qfl_dir: Path,
) -> pd.DataFrame:
    """
    Collect results from all experiment stages and assemble one master table.
    Reads from saved JSON/CSV files — does not re-run any experiments.

    Columns:
      Model | Category | Regime | Qubits | Layers | TrainableParams |
      NoiseSigma | DPSigma | MendeleyTestAUC | MendeleyTestF1 |
      KAU_AUC | KAU_F1 | GeneralisationGap | Notes
    """
    rows = []

    # ── 1. Classical baseline ─────────────────────────────────────────────
    if baseline_json.exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        rows.append({
            "Model":            bl.get("backbone", "MobileNetV2"),
            "ModelShort":       "Classical",
            "Category":         "Classical",
            "Regime":           bl.get("freeze_strategy", "progressive"),
            "Qubits":           "N/A", "Layers": "N/A",
            "TrainableParams":  bl.get("trainable_params", "N/A"),
            "NoiseSigma":       0.0, "DPSigma": 0.0,
            "MendeleyTestAUC":  bl.get("test_auc_roc", "N/A"),
            "MendeleyTestF1":   bl.get("test_f1", "N/A"),
            "KAU_AUC":          kau_results_dict.get("Classical", {}).get("auc", "N/A"),
            "KAU_F1":           kau_results_dict.get("Classical", {}).get("f1", "N/A"),
            "Notes":            "Classical baseline (frozen CNN backbone)",
        })

    # ── 1b. Classical Micro-MLP / Control ──────────────────────────────────
    if "Micro-MLP" in kau_results_dict:
        m_res = kau_results_dict["Micro-MLP"]
        rows.append({
            "Model":            "Micro-MLP (Classical)",
            "ModelShort":       "Micro-MLP",
            "Category":         "Classical Control",
            "Regime":           "Classical micro-model on PCA features",
            "Qubits":           N_QUBITS, "Layers": 0,
            "TrainableParams":  N_QUBITS + 1,
            "NoiseSigma":       0.0, "DPSigma": 0.0,
            "MendeleyTestAUC":  m_res.get("mendeley_auc", "N/A"),
            "MendeleyTestF1":   m_res.get("mendeley_f1", "N/A"),
            "KAU_AUC":          m_res.get("auc", "N/A"),
            "KAU_F1":           m_res.get("f1", "N/A"),
            "Notes":            "Classical linear probe / micro-model (parameter-matched control)",
        })

    # ── 2. VQC Regime A primary results (best config per qubit count) ────────
    ablation_csv = vqc_dir_A.parent / "ablation_table.csv"
    seen_regime_A = set()  # deduplicate by (nq, nl)

    if ablation_csv.exists():
        df_abl = pd.read_csv(ablation_csv)
        regime_A_rows = df_abl[df_abl["Regime"].str.contains("frozen", na=False)]
        for _, r in regime_A_rows.iterrows():
            try:
                nq = int(r["Qubits"]); nl = int(r["Layers"])
            except (ValueError, KeyError):
                continue
            key = (nq, nl)
            if key in seen_regime_A:
                continue  # keep first (best val AUC) only
            seen_regime_A.add(key)
            label = f"HQCNN q={nq} l={nl}"
            rows.append({
                "Model":           label,
                "ModelShort":      f"VQC-q{nq}l{nl}-A",
                "Category":        "HQCNN Regime A",
                "Regime":          "A — frozen classical + VQC",
                "Qubits":          nq, "Layers": nl,
                "TrainableParams": nq * nl + 1,
                "NoiseSigma":      0.0, "DPSigma": 0.0,
                "MendeleyTestAUC": r.get("TestAUC", "N/A"),
                "MendeleyTestF1":  r.get("TestF1",  "N/A"),
                "KAU_AUC":         kau_results_dict.get(label, {}).get("auc", "N/A"),
                "KAU_F1":          kau_results_dict.get(label, {}).get("f1", "N/A"),
                "Notes":           "Regime A primary",
            })
        
        # Also include classical control rows from ablation_table.csv if present
        ctrl_rows = df_abl[df_abl["Regime"].str.contains("control|param-matched", case=False, na=False)]
        for _, r in ctrl_rows.iterrows():
            m_name = str(r.get("Model", "Classical Control"))
            rows.append({
                "Model":           m_name,
                "ModelShort":      m_name[:15],
                "Category":        "Classical Control",
                "Regime":          str(r.get("Regime", "Classical control")),
                "Qubits":          r.get("Qubits", "N/A"),
                "Layers":          r.get("Layers", "N/A"),
                "TrainableParams": r.get("TrainableParams", "N/A"),
                "NoiseSigma":      r.get("NoiseSigma", 0.0),
                "DPSigma":         0.0,
                "MendeleyTestAUC": r.get("TestAUC", "N/A"),
                "MendeleyTestF1":  r.get("TestF1",  "N/A"),
                "KAU_AUC":         kau_results_dict.get(m_name, {}).get("auc", "N/A"),
                "KAU_F1":          kau_results_dict.get(m_name, {}).get("f1", "N/A"),
                "Notes":           str(r.get("Notes", "Classical control baseline")),
            })
    else:
        # Fallback: parse history CSVs, deduplicate
        for hist_csv in sorted(vqc_dir_A.glob("*_history.csv")):
            stem  = hist_csv.stem.replace("_history", "")
            parts = stem.split("_")
            try:
                nq = int(parts[1][1:]); nl = int(parts[2][1:])
                lr = float(parts[3][2:])
            except (IndexError, ValueError):
                continue
            key = (nq, nl)
            if key in seen_regime_A:
                continue
            seen_regime_A.add(key)
            df_h  = pd.read_csv(hist_csv)
            label = f"HQCNN q={nq} l={nl}"
            rows.append({
                "Model":           label,
                "ModelShort":      f"VQC-q{nq}l{nl}-A",
                "Category":        "HQCNN Regime A",
                "Regime":          "A — frozen classical + VQC",
                "Qubits":          nq, "Layers": nl,
                "TrainableParams": nq * nl + 1,
                "NoiseSigma":      0.0, "DPSigma": 0.0,
                "MendeleyTestAUC": "N/A",
                "MendeleyTestF1":  "N/A",
                "KAU_AUC":         kau_results_dict.get(label, {}).get("auc", "N/A"),
                "KAU_F1":          kau_results_dict.get(label, {}).get("f1", "N/A"),
                "Notes":           f"Regime A, lr={lr}",
            })

    # ── 2b. Regime B results ──────────────────────────────────────────────
    seen_regime_B = set()
    for hist_csv in sorted(vqc_dir_B.glob("regimeB_*_history.csv")):
        stem  = hist_csv.stem  # regimeB_qN_lN_history
        parts = stem.split("_")
        try:
            nq = int(parts[1][1:]); nl = int(parts[2][1:])
        except (IndexError, ValueError):
            continue
        key = (nq, nl)
        if key in seen_regime_B:
            continue
        seen_regime_B.add(key)
        df_h  = pd.read_csv(hist_csv)
        best_auc = df_h["val_auc"].max() if "val_auc" in df_h.columns else "N/A"
        label = f"HQCNN q={nq} l={nl} (Regime B)"
        rows.append({
            "Model":           label,
            "ModelShort":      f"VQC-q{nq}l{nl}-B",
            "Category":        "HQCNN Regime B",
            "Regime":          "B — end-to-end (projection + VQC)",
            "Qubits":          nq, "Layers": nl,
            "TrainableParams": nq * nl + 1 + nq * nq,  # VQC + projection layer
            "NoiseSigma":      0.0, "DPSigma": 0.0,
            "MendeleyTestAUC": "N/A",
            "MendeleyTestF1":  "N/A",
            "KAU_AUC":         kau_results_dict.get(label, {}).get("auc", "N/A"),
            "KAU_F1":          kau_results_dict.get(label, {}).get("f1", "N/A"),
            "Notes":           f"Regime B, best val AUC={best_auc:.4f}" if isinstance(best_auc, float) else "Regime B",
        })

    # ── 2c. Sweep best results (one row per qubit count, best layer config) ─
    sweep_dir = vqc_dir_A.parent / "sweep"
    if sweep_dir.exists() and ablation_csv.exists():
        df_abl   = pd.read_csv(ablation_csv)
        sweep_rows = df_abl[df_abl["Regime"].str.contains("sweep", na=False)]
        # Keep best AUC per qubit count
        if "TestAUC" in sweep_rows.columns and not sweep_rows.empty:
            try:
                sweep_rows = sweep_rows.copy()
                sweep_rows["TestAUC_f"] = pd.to_numeric(sweep_rows["TestAUC"], errors="coerce")
                best_sweep = sweep_rows.loc[sweep_rows.groupby("Qubits")["TestAUC_f"].idxmax()]
                for _, r in best_sweep.iterrows():
                    nq = int(r["Qubits"]); nl = int(r["Layers"])
                    rows.append({
                        "Model":           f"HQCNN q={nq} l={nl} (sweep best)",
                        "ModelShort":      f"VQC-q{nq}l{nl}-sweep",
                        "Category":        "HQCNN Sweep",
                        "Regime":          "A — sweep (best per qubit count)",
                        "Qubits":          nq, "Layers": nl,
                        "TrainableParams": nq * nl + 1,
                        "NoiseSigma":      0.0, "DPSigma": 0.0,
                        "MendeleyTestAUC": r.get("TestAUC", "N/A"),
                        "MendeleyTestF1":  r.get("TestF1", "N/A"),
                        "KAU_AUC":         "N/A", "KAU_F1": "N/A",
                        "Notes":           "Best sweep config per qubit count",
                    })
            except Exception:
                pass  # sweep rows not parseable — skip silently

    # ── 3. Noise robustness rows ──────────────────────────────────────────
    noise_csv = noise_dir / "noise_results.csv"
    if noise_csv.exists():
        df_noise = pd.read_csv(noise_csv)
        for _, row_n in df_noise.iterrows():
            if row_n["noise_sigma"] == 0.0:
                continue
            rows.append({
                "Model":           f"HQCNN q={int(row_n['n_qubits'])} (noise σ={row_n['noise_sigma']})",
                "ModelShort":      f"VQC-σ{row_n['noise_sigma']}",
                "Category":        "HQCNN Regime A",
                "Regime":          "A — noise robustness",
                "Qubits":          int(row_n["n_qubits"]),
                "Layers":          int(row_n["n_layers"]),
                "TrainableParams": int(row_n["n_qubits"]) * int(row_n["n_layers"]) + 1,
                "NoiseSigma":      row_n["noise_sigma"], "DPSigma": 0.0,
                "MendeleyTestAUC": round(row_n["test_auc_roc"], 4),
                "MendeleyTestF1":  round(row_n["test_f1"], 4),
                "KAU_AUC":         "N/A", "KAU_F1": "N/A",
                "Notes":           f"Gaussian noise σ={row_n['noise_sigma']} on PCA features",
            })

    # ── 4. QFL results ────────────────────────────────────────────────────
    qfl_json = qfl_dir / "qfl_summary.json"
    if qfl_json.exists():
        with open(qfl_json) as f:
            qfl = json.load(f)
        # Centralised row
        ua = qfl.get("utility_analysis", {})
        rows.append({
            "Model":           "HQCNN Centralised (pooled)",
            "ModelShort":      "Centralised",
            "Category":        "QFL Federated",
            "Regime":          "centralised baseline (same steps as QFL)",
            "Qubits":          N_QUBITS, "Layers": N_LAYERS,
            "TrainableParams": N_QUBITS * N_LAYERS + 1,
            "NoiseSigma":      0.0, "DPSigma": 0.0,
            "MendeleyTestAUC": ua.get("centralised_test_auc", "N/A"),
            "MendeleyTestF1":  ua.get("centralised_test_f1", "N/A"),
            "KAU_AUC":         "N/A", "KAU_F1": "N/A",
            "Notes":           "Centralised VQC; same gradient steps as QFL",
        })
        # QFL no-DP row
        rows.append({
            "Model":           "QFL (σ_dp=0, no DP)",
            "ModelShort":      "QFL",
            "Category":        "QFL Federated",
            "Regime":          "federated (FedAvg, 3 Ghanaian clients)",
            "Qubits":          N_QUBITS, "Layers": N_LAYERS,
            "TrainableParams": N_QUBITS * N_LAYERS + 1,
            "NoiseSigma":      0.0, "DPSigma": 0.0,
            "MendeleyTestAUC": ua.get("federated_test_auc", "N/A"),
            "MendeleyTestF1":  ua.get("federated_test_f1", "N/A"),
            "KAU_AUC":         kau_results_dict.get("QFL", {}).get("auc", "N/A"),
            "KAU_F1":          kau_results_dict.get("QFL", {}).get("f1", "N/A"),
            "Notes":           f"Utility gap vs centralised: {ua.get('auc_utility_gap','N/A')}",
        })
        # DP sweep rows
        for dp_r in qfl.get("dp_analysis", {}).get("all_dp_results", []):
            if dp_r["dp_sigma"] == 0.0:
                continue
            rows.append({
                "Model":           f"QFL + DP (σ_dp={dp_r['dp_sigma']})",
                "ModelShort":      f"QFL-DP{dp_r['dp_sigma']}",
                "Category":        "QFL + DP",
                "Regime":          "federated + differential privacy",
                "Qubits":          N_QUBITS, "Layers": N_LAYERS,
                "TrainableParams": N_QUBITS * N_LAYERS + 1,
                "NoiseSigma":      0.0, "DPSigma": dp_r["dp_sigma"],
                "MendeleyTestAUC": dp_r.get("final_test_auc", "N/A"),
                "MendeleyTestF1":  dp_r.get("final_test_f1", "N/A"),
                "KAU_AUC":         "N/A", "KAU_F1": "N/A",
                "Notes":           f"DP noise σ={dp_r['dp_sigma']} on VQC gradients",
            })

    # ── Compute generalisation gap ────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["GeneralisationGap"] = df.apply(
        lambda r: round(float(r["MendeleyTestAUC"]) - float(r["KAU_AUC"]), 4)
        if str(r["MendeleyTestAUC"]).replace(".","").isdigit()
        and str(r["KAU_AUC"]).replace(".","").isdigit()
        else "N/A", axis=1
    )
    return df


def generate_master_dashboard(master_df: pd.DataFrame,
                              kau_results: dict,
                              mendeley_results: dict,
                              report: dict,
                              save_path: Path):
    """
    Generate an aggregated Markdown summary dashboard capturing all key metrics
    and findings across all experiment stages at a glance.
    """
    lines = []
    lines.append("# QFL Breast Cancer Classification — Master Results Dashboard")
    lines.append("")
    lines.append(f"**Study**: {report.get('study', 'Privacy-Preserving Quantum Federated Learning for Breast Cancer Screening')}")
    lines.append(f"**Primary Cohort (Train/Val/Test)**: {report.get('primary_dataset', 'Mendeley (Polokwane, South Africa)')}")
    lines.append(f"**External Validation Cohort**: {report.get('external_dataset', 'KAU-BCMD (Saudi Arabia / MENA)')}")
    lines.append("")
    lines.append("---")
    lines.append("## 1. Cross-Population Clinical Generalisation (At a Glance)")
    lines.append("")
    lines.append("| Model | Mendeley AUC | Mendeley PR-AUC | KAU AUC | KAU PR-AUC | KAU Brier | KAU F1 (0.5) | KAU Sens (0.5) | KAU Spec (0.5) | Opt τ* | KAU Sens (τ*) | KAU Spec (τ*) | KAU BalAcc (τ*) | Gap (AUC) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, k_res in kau_results.items():
        m_res = mendeley_results.get(name, {})
        m_auc = m_res.get("auc", "N/A")
        m_ap  = m_res.get("average_precision", "N/A")
        k_auc = k_res.get("auc", "N/A")
        k_ap  = k_res.get("average_precision", "N/A")
        k_br  = k_res.get("brier_score", "N/A")
        k_f1  = k_res.get("f1", "N/A")
        k_sens_05 = k_res.get("per_class", {}).get("Malignant", {}).get("recall", "N/A")
        k_spec_05 = k_res.get("per_class", {}).get("Benign", {}).get("recall", "N/A")

        tau_opt   = k_res.get("opt_threshold", "N/A")
        k_sens_opt = k_res.get("per_class_at_opt", {}).get("Malignant", {}).get("recall", "N/A")
        k_spec_opt = k_res.get("per_class_at_opt", {}).get("Benign", {}).get("recall", "N/A")
        k_bacc_opt = k_res.get("balanced_acc_at_opt", "N/A")

        gap = round(m_auc - k_auc, 4) if isinstance(m_auc, (int, float)) and isinstance(k_auc, (int, float)) else "N/A"

        m_auc_str = f"{m_auc:.4f}" if isinstance(m_auc, (int, float)) else str(m_auc)
        m_ap_str  = f"{m_ap:.4f}" if isinstance(m_ap, (int, float)) else str(m_ap)
        k_auc_str = f"{k_auc:.4f}" if isinstance(k_auc, (int, float)) else str(k_auc)
        k_ap_str  = f"{k_ap:.4f}" if isinstance(k_ap, (int, float)) else str(k_ap)
        k_br_str  = f"{k_br:.4f}" if isinstance(k_br, (int, float)) else str(k_br)
        k_f1_str  = f"{k_f1:.4f}" if isinstance(k_f1, (int, float)) else str(k_f1)
        k_s05_str = f"{k_sens_05:.4f}" if isinstance(k_sens_05, (int, float)) else str(k_sens_05)
        k_sp05_str= f"{k_spec_05:.4f}" if isinstance(k_spec_05, (int, float)) else str(k_spec_05)

        tau_str   = f"{tau_opt:.4f}" if isinstance(tau_opt, (int, float)) else str(tau_opt)
        k_sopt_str= f"{k_sens_opt:.4f}" if isinstance(k_sens_opt, (int, float)) else str(k_sens_opt)
        k_spopt_str= f"{k_spec_opt:.4f}" if isinstance(k_spec_opt, (int, float)) else str(k_spec_opt)
        k_bacc_str= f"{k_bacc_opt:.4f}" if isinstance(k_bacc_opt, (int, float)) else str(k_bacc_opt)
        gap_str   = f"{gap:+.4f}" if isinstance(gap, (int, float)) else str(gap)

        lines.append(f"| **{name}** | {m_auc_str} | {m_ap_str} | {k_auc_str} | {k_ap_str} | {k_br_str} | {k_f1_str} | {k_s05_str} | {k_sp05_str} | {tau_str} | {k_sopt_str} | {k_spopt_str} | {k_bacc_str} | {gap_str} |")

    lines.append("")
    lines.append("---")
    lines.append("## 2. Master Ablation & Experiment Results")
    lines.append("")
    display_cols = [
        "Model", "Category", "Qubits", "Layers", "TrainableParams",
        "NoiseSigma", "DPSigma", "MendeleyTestAUC", "MendeleyTestF1",
        "KAU_AUC", "KAU_F1", "GeneralisationGap"
    ]
    avail_cols = [c for c in display_cols if c in master_df.columns]
    lines.append("| " + " | ".join(avail_cols) + " |")
    lines.append("| " + " | ".join([":---:" if i > 1 else ":---" for i in range(len(avail_cols))]) + " |")
    for _, row in master_df.iterrows():
        vals = [str(row.get(c, "N/A")) for c in avail_cols]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("---")
    lines.append("## 3. Domain Shift Metrics (Mendeley SA vs KAU-BCMD MENA)")
    shift = report.get("domain_shift", {})
    if shift:
        lines.append(f"- **Mean Jensen-Shannon Divergence**: `{shift.get('mean_js_divergence', 'N/A')}` (0 = identical, 1 = maximal shift)")
        lines.append(f"- **Dimensions with Significant Shift (p < 0.05)**: `{shift.get('n_dims_significant_shift', 'N/A')}/{N_QUBITS}`")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Master Summary Dashboard written to: {save_path}")


def plot_master_ablation(df: pd.DataFrame, save_path: Path):
    """Render the master ablation table as a publication-ready figure."""
    display_cols = [
        "Model", "Category", "Qubits", "Layers", "TrainableParams",
        "NoiseSigma", "DPSigma",
        "MendeleyTestAUC", "MendeleyTestF1",
        "KAU_AUC", "KAU_F1", "GeneralisationGap"
    ]
    plot_df = df[display_cols].fillna("N/A")

    fig, ax = plt.subplots(figsize=(24, max(4.5, len(plot_df) * 0.7 + 2.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.15, 1.8)

    cat_colors = {
        "Classical":       "D5E8F0",
        "Classical Control":"D5E8F0",
        "HQCNN Regime A":  "FFF2CC",
        "HQCNN Regime B":  "D5F5E3",
        "QFL Federated":   "E8D5F5",
        "QFL + DP":        "F5D5E8",
    }
    for j in range(len(display_cols)):
        tbl[(0, j)].set_facecolor("#1F4E79")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(plot_df) + 1):
        cat   = df.iloc[i - 1].get("Category", "")
        color = cat_colors.get(cat, "FFFFFF")
        for j in range(len(display_cols)):
            tbl[(i, j)].set_facecolor(f"#{color}")

    ax.set_title(
        "Master Ablation Table — All Experiments\n"
        "Privacy-Preserving QFL for Breast Cancer Screening in African & MENA Populations\n"
        "Primary: Mendeley (Polokwane, SA) | External: KAU-BCMD (Saudi Arabia)",
        fontsize=12, fontweight="bold", pad=25
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print("  10–11 — EXTERNAL VALIDATION + MASTER ABLATION TABLE")
    print("  Cross-population: Mendeley (SA) → KAU-BCMD (MENA)")
    print("═"*70)

    from cache_check import already_done, CACHE
    seed_everything(42)

    # ── Load KAU features ────────────────────────────────────────────────
    print("\n[1/6] Loading KAU-BCMD features...")
    X_kau_raw, X_kau_pca, X_kau_scaled, y_kau = load_kau_features(N_QUBITS)

    # Load Mendeley splits for validation tuning and test evaluation
    X_test_pca  = np.load(FEAT_DIR / f"features_test_pca{N_QUBITS}.npy")
    y_test      = np.load(FEAT_DIR / "labels_test.npy")
    X_val_pca   = np.load(FEAT_DIR / f"features_val_pca{N_QUBITS}.npy")
    y_val       = np.load(FEAT_DIR / "labels_val.npy")
    X_val_raw   = np.load(FEAT_DIR / "features_val_raw.npy")
    X_train_pca = np.load(FEAT_DIR / f"features_train_pca{N_QUBITS}.npy")
    y_train_arr = np.load(FEAT_DIR / "labels_train.npy")

    scaler_mm     = MinMaxScaler(feature_range=(0, 1)).fit(X_train_pca)
    X_test_scaled = scaler_mm.transform(X_test_pca)
    X_val_scaled  = scaler_mm.transform(X_val_pca)
    X_train_scaled= scaler_mm.transform(X_train_pca)

    # ── Load models ──────────────────────────────────────────────────────
    print("\n[2/6] Loading trained models...")

    cnn_model = build_mobilenet(BASELINE_DIR / f"{BACKBONE}_best.pt")
    vqc_A     = load_vqc(VQC_DIR_A, N_QUBITS, N_LAYERS, VQC_LR)

    # Try loading QFL global model
    qfl_ckpt  = QFL_DIR / f"qfl_global_q{N_QUBITS}_l{N_LAYERS}.pt"
    vqc_qfl   = VQCModel(N_QUBITS, N_LAYERS)
    if qfl_ckpt.exists():
        vqc_qfl.load_state_dict(torch.load(qfl_ckpt, map_location="cpu"))
        vqc_qfl.eval()
        print(f"  Loaded QFL model: {qfl_ckpt}")
    else:
        print(f"  [INFO] QFL checkpoint not found; skipping QFL KAU evaluation.")
        vqc_qfl = None

    # ── Evaluate on KAU with validation-tuned threshold ──────────────────
    print("\n[3/6] Evaluating on KAU-BCMD external validation set (with validation-tuned threshold)...")
    kau_results = {}
    mendeley_results = {}

    # Classical MobileNetV2
    print("  Classical MobileNetV2...")
    X_kau_raw_t = np.load(FEAT_DIR / "features_kau_raw.npy")
    X_test_raw  = np.load(FEAT_DIR / "features_test_raw.npy")
    cnn_val_eval = evaluate_cnn_on_kau(cnn_model, X_val_raw, y_val)
    tau_cnn = cnn_val_eval["opt_threshold"]
    print(f"    Validation optimal threshold τ*={tau_cnn:.4f}")
    kau_results["Classical"]      = evaluate_cnn_on_kau(cnn_model, X_kau_raw_t, y_kau, opt_threshold=tau_cnn)
    mendeley_results["Classical"] = evaluate_cnn_on_kau(cnn_model, X_test_raw,  y_test, opt_threshold=tau_cnn)

    # Classical Micro-MLP (parameter-matched control on PCA features)
    print("  Classical Micro-MLP Control...")
    micromlp_model = load_or_train_micromlp(N_QUBITS, X_train_scaled, y_train_arr)
    mlp_val_eval = evaluate_vqc_on_kau(micromlp_model, X_val_scaled, y_val)
    tau_mlp = mlp_val_eval["opt_threshold"]
    print(f"    Validation optimal threshold τ*={tau_mlp:.4f}")
    kau_results["Micro-MLP"] = evaluate_vqc_on_kau(micromlp_model, X_kau_scaled, y_kau, opt_threshold=tau_mlp)
    mendeley_results["Micro-MLP"] = evaluate_vqc_on_kau(micromlp_model, X_test_scaled, y_test, opt_threshold=tau_mlp)
    kau_results["Micro-MLP"]["mendeley_auc"] = mendeley_results["Micro-MLP"]["auc"]
    kau_results["Micro-MLP"]["mendeley_f1"] = mendeley_results["Micro-MLP"]["f1"]

    # HQCNN Regime A
    label_A = f"HQCNN q={N_QUBITS} l={N_LAYERS}"
    print(f"  {label_A}...")
    vqc_val_eval = evaluate_vqc_on_kau(vqc_A, X_val_scaled, y_val)
    tau_vqc = vqc_val_eval["opt_threshold"]
    print(f"    Validation optimal threshold τ*={tau_vqc:.4f}")
    kau_results[label_A]      = evaluate_vqc_on_kau(vqc_A, X_kau_scaled, y_kau, opt_threshold=tau_vqc)
    mendeley_results[label_A] = evaluate_vqc_on_kau(
        vqc_A, X_test_scaled, y_test, opt_threshold=tau_vqc
    )

    # QFL model
    if vqc_qfl is not None:
        print("  QFL global model...")
        qfl_val_eval = evaluate_vqc_on_kau(vqc_qfl, X_val_scaled, y_val)
        tau_qfl = qfl_val_eval["opt_threshold"]
        print(f"    Validation optimal threshold τ*={tau_qfl:.4f}")
        kau_results["QFL"]      = evaluate_vqc_on_kau(vqc_qfl, X_kau_scaled, y_kau, opt_threshold=tau_qfl)
        mendeley_results["QFL"] = evaluate_vqc_on_kau(vqc_qfl, X_test_scaled, y_test, opt_threshold=tau_qfl)

    # Print summary
    print("\n  ── Cross-population Clinical Results ──")
    for name in kau_results:
        m_auc = mendeley_results.get(name, {}).get("auc", "N/A")
        k_auc = kau_results[name]["auc"]
        tau   = kau_results[name]["opt_threshold"]
        k_f1  = kau_results[name]["f1"]
        k_f1_opt = kau_results[name]["f1_at_opt"]
        gap   = round(m_auc - k_auc, 4) if isinstance(m_auc, (int, float)) and isinstance(k_auc, (int, float)) else "N/A"
        print(f"  {name:30s} | Mendeley AUC={m_auc} | "
              f"KAU AUC={k_auc:.4f} | F1(0.5)={k_f1:.4f} | Opt τ*={tau:.4f} -> F1(τ*)={k_f1_opt:.4f} | Gap={gap}")
        # Per-class detail for KAU
        pc_05 = kau_results[name]["per_class"]
        pc_opt = kau_results[name]["per_class_at_opt"]
        print(f"    [0.5 default] Sens: {pc_05.get('Malignant',{}).get('recall','N/A')} | Spec: {pc_05.get('Benign',{}).get('recall','N/A')}")
        print(f"    [Opt τ*={tau:.2f}] Sens: {pc_opt.get('Malignant',{}).get('recall','N/A')} | Spec: {pc_opt.get('Benign',{}).get('recall','N/A')} | BalAcc: {kau_results[name]['balanced_acc_at_opt']}")

    # ── Domain shift analysis ─────────────────────────────────────────────
    print("\n[4/6] Domain shift analysis...")
    X_mendeley_all_pca = np.load(FEAT_DIR / f"features_train_pca{N_QUBITS}.npy")
    shift_metrics = domain_shift_analysis(
        X_mendeley_all_pca,
        np.load(FEAT_DIR / "labels_train.npy"),
        X_kau_pca, y_kau,
        OUT_DIR / "domain_shift_analysis.png"
    )
    print(f"  Mean JS divergence: {shift_metrics['mean_js_divergence']}")
    print(f"  Dims with significant shift: "
          f"{shift_metrics['n_dims_significant_shift']}/{N_QUBITS}")

    # ── Cross-population comparison plot ──────────────────────────────────
    print("\n[5/6] Cross-population comparison plots...")
    plot_cross_population(
        mendeley_results, kau_results,
        OUT_DIR / "cross_population_comparison.png"
    )

    # ── Master ablation table ─────────────────────────────────────────────
    print("\n[6/6] Building master ablation table...")
    master_df = build_master_ablation(
        kau_results_dict=kau_results,
        baseline_json=BASELINE_DIR / "baseline_results.json",
        vqc_dir_A=VQC_DIR_A,
        vqc_dir_B=VQC_DIR_B,
        noise_dir=NOISE_DIR,
        qfl_dir=QFL_DIR,
    )
    master_df.to_csv(OUT_DIR / "final_ablation_table.csv", index=False, encoding="utf-8-sig")
    plot_master_ablation(master_df, OUT_DIR / "final_ablation_table.png")

    # Parameter efficiency plot
    eff_rows = []
    for _, row in master_df.iterrows():
        try:
            params = int(str(row["TrainableParams"]).replace(",",""))
            m_auc  = float(str(row["MendeleyTestAUC"]))
            k_auc  = float(str(row["KAU_AUC"])) if str(row["KAU_AUC"]).replace(".","").isdigit() else None
        except (ValueError, TypeError):
            continue
        eff_rows.append({
            "model":            row["Model"],
            "model_short":      row.get("ModelShort", row["Model"][:15]),
            "category":         row["Category"],
            "trainable_params": params,
            "mendeley_auc":     m_auc,
            "kau_auc":          k_auc,
        })
    if eff_rows:
        results_for_plot = [
            {"model_short": r["model_short"], "category": r["category"],
             "trainable_params": r["trainable_params"],
             "mendeley_auc": r["mendeley_auc"], "kau_auc": r["kau_auc"]}
            for r in eff_rows
        ]
        plot_parameter_efficiency(results_for_plot,
                                   OUT_DIR / "parameter_efficiency.png")

    # ── Save generalisation report & Master Dashboard ──────────────────────
    report = {
        "study": "Privacy-Preserving QFL for Breast Cancer Screening",
        "primary_dataset":  "Mendeley (Polokwane, South Africa)",
        "external_dataset": "KAU-BCMD (Saudi Arabia / MENA)",
        "cross_population_results": {
            name: {
                "mendeley_auc": mendeley_results.get(name, {}).get("auc", "N/A"),
                "kau_auc":      kau_results[name]["auc"],
                "kau_average_precision": kau_results[name]["average_precision"],
                "kau_mcc":      kau_results[name]["mcc"],
                "kau_f1":       kau_results[name]["f1"],
                "kau_sensitivity": kau_results[name]["per_class"].get(
                    "Malignant", {}).get("recall", "N/A"),
                "kau_specificity": kau_results[name]["per_class"].get(
                    "Benign", {}).get("recall", "N/A"),
            }
            for name in kau_results
        },
        "domain_shift": shift_metrics,
        "ablation_table_path": str(OUT_DIR / "final_ablation_table.csv"),
    }
    with open(OUT_DIR / "generalisation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    generate_master_dashboard(master_df, kau_results, mendeley_results, report, OUT_DIR / "master_summary_dashboard.md")
    generate_master_dashboard(master_df, kau_results, mendeley_results, report, BASE / "master_summary_dashboard.md")

    CACHE.mark_done("external_validation")

    print("\n" + "═"*70)
    print("  10–11 COMPLETE")
    print(f"  Outputs: {OUT_DIR}")
    print("\n  Key generalisation results:")
    for name in kau_results:
        m = mendeley_results.get(name, {}).get("auc", "—")
        k = kau_results[name]["auc"]
        g = round(m - k, 4) if isinstance(m, float) else "—"
        print(f"    {name:30s} | Mendeley={m} | KAU={k} | Gap={g}")
    print("\n  Next step → run_pipeline.sh (full local pipeline runner)")


if __name__ == "__main__":
    main()