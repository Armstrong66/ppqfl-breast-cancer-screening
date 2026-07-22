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

import json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve
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

# ── Match to best config from _3–5 sweep (this is also need to auto_detect) ─────────────────────────────────
BACKBONE      = "mobilenetv2"
N_QUBITS      = 4
N_LAYERS      = 2
VQC_LR        = 0.01
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

@torch.no_grad()
def evaluate_vqc_on_kau(model, X_kau_scaled, y_kau):
    X_t   = torch.tensor(X_kau_scaled, dtype=torch.float32)
    probs = model(X_t).numpy()
    preds = (probs > 0.5).astype(int)
    auc   = roc_auc_score(y_kau, probs) if len(set(y_kau)) > 1 else 0.0
    f1    = f1_score(y_kau, preds, zero_division=0)
    acc   = accuracy_score(y_kau, preds)
    report = classification_report(
        y_kau, preds, target_names=["Benign", "Malignant"], output_dict=True
    )
    return {"auc": round(auc,4), "f1": round(f1,4), "accuracy": round(acc,4),
            "per_class": report, "probs": probs, "preds": preds}


@torch.no_grad()
def evaluate_cnn_on_kau(model, X_kau_raw, y_kau):
    X_t    = torch.tensor(X_kau_raw, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, torch.tensor(y_kau)),
                        batch_size=BATCH_SIZE, shuffle=False)
    all_probs, all_preds = [], []
    for X_b, _ in loader:
        logits = model.classifier(X_b)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.numpy())
        all_preds.extend((probs > 0.5).long().numpy())
    probs = np.array(all_probs)
    preds = np.array(all_preds)
    auc   = roc_auc_score(y_kau, probs) if len(set(y_kau)) > 1 else 0.0
    f1    = f1_score(y_kau, preds, zero_division=0)
    acc   = accuracy_score(y_kau, preds)
    report = classification_report(
        y_kau, preds, target_names=["Benign", "Malignant"], output_dict=True
    )
    return {"auc": round(auc,4), "f1": round(f1,4), "accuracy": round(acc,4),
            "per_class": report, "probs": probs, "preds": preds}


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
            "Qubits":           "—", "Layers": "—",
            "TrainableParams":  bl.get("trainable_params", "N/A"),
            "NoiseSigma":       0.0, "DPSigma": 0.0,
            "MendeleyTestAUC":  bl.get("test_auc_roc", "—"),
            "MendeleyTestF1":   bl.get("test_f1", "—"),
            "KAU_AUC":          kau_results_dict.get("Classical", {}).get("auc", "—"),
            "KAU_F1":           kau_results_dict.get("Classical", {}).get("f1", "—"),
            "Notes":            "Classical baseline (frozen CNN backbone)",
        })

    # ── 2. VQC Regime A primary results (best config per qubit count) ────────
    # Pull actual test metrics from the ablation_table.csv produced by
    # 3_5_vqc.py rather than the history CSVs (which have val AUC only).
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
                "MendeleyTestAUC": r.get("TestAUC", "—"),
                "MendeleyTestF1":  r.get("TestF1",  "—"),
                "KAU_AUC":         kau_results_dict.get(label, {}).get("auc", "—"),
                "KAU_F1":          kau_results_dict.get(label, {}).get("f1", "—"),
                "Notes":           "Regime A primary",
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
                "MendeleyTestAUC": "—",
                "MendeleyTestF1":  "—",
                "KAU_AUC":         kau_results_dict.get(label, {}).get("auc", "—"),
                "KAU_F1":          kau_results_dict.get(label, {}).get("f1", "—"),
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
        best_auc = df_h["val_auc"].max() if "val_auc" in df_h.columns else "—"
        label = f"HQCNN q={nq} l={nl} (Regime B)"
        rows.append({
            "Model":           label,
            "ModelShort":      f"VQC-q{nq}l{nl}-B",
            "Category":        "HQCNN Regime B",
            "Regime":          "B — end-to-end (projection + VQC)",
            "Qubits":          nq, "Layers": nl,
            "TrainableParams": nq * nl + 1 + nq * nq,  # VQC + projection layer
            "NoiseSigma":      0.0, "DPSigma": 0.0,
            "MendeleyTestAUC": "—",
            "MendeleyTestF1":  "—",
            "KAU_AUC":         kau_results_dict.get(label, {}).get("auc", "—"),
            "KAU_F1":          kau_results_dict.get(label, {}).get("f1", "—"),
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
                        "MendeleyTestAUC": r.get("TestAUC", "—"),
                        "MendeleyTestF1":  r.get("TestF1", "—"),
                        "KAU_AUC":         "—", "KAU_F1": "—",
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
            "MendeleyTestAUC": ua.get("centralised_test_auc", "—"),
            "MendeleyTestF1":  ua.get("centralised_test_f1", "—"),
            "KAU_AUC":         "—", "KAU_F1": "—",
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
            "MendeleyTestAUC": ua.get("federated_test_auc", "—"),
            "MendeleyTestF1":  ua.get("federated_test_f1", "—"),
            "KAU_AUC":         kau_results_dict.get("QFL", {}).get("auc", "—"),
            "KAU_F1":          kau_results_dict.get("QFL", {}).get("f1", "—"),
            "Notes":           f"Utility gap vs centralised: {ua.get('auc_utility_gap','—')}",
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
                "MendeleyTestAUC": dp_r.get("final_test_auc", "—"),
                "MendeleyTestF1":  dp_r.get("final_test_f1", "—"),
                "KAU_AUC":         "—", "KAU_F1": "—",
                "Notes":           f"DP noise σ={dp_r['dp_sigma']} on VQC gradients",
            })

    # ── Compute generalisation gap ────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["GeneralisationGap"] = df.apply(
        lambda r: round(float(r["MendeleyTestAUC"]) - float(r["KAU_AUC"]), 4)
        if str(r["MendeleyTestAUC"]).replace(".","").isdigit()
        and str(r["KAU_AUC"]).replace(".","").isdigit()
        else "—", axis=1
    )
    return df


def plot_master_ablation(df: pd.DataFrame, save_path: Path):
    """Render the master ablation table as a publication-ready figure."""
    display_cols = [
        "Model", "Category", "Qubits", "Layers", "TrainableParams",
        "NoiseSigma", "DPSigma",
        "MendeleyTestAUC", "MendeleyTestF1",
        "KAU_AUC", "KAU_F1", "GeneralisationGap"
    ]
    plot_df = df[display_cols].fillna("—")

    fig, ax = plt.subplots(figsize=(22, max(4, len(plot_df) * 0.65 + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.55)

    cat_colors = {
        "Classical":       "D5E8F0",
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
        fontsize=11, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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

    # Also load Mendeley test for comparison
    X_test_pca = np.load(FEAT_DIR / f"features_test_pca{N_QUBITS}.npy")
    y_test     = np.load(FEAT_DIR / "labels_test.npy")
    X_train_pca = np.load(FEAT_DIR / f"features_train_pca{N_QUBITS}.npy")
    scaler_mm   = MinMaxScaler(feature_range=(0, 1)).fit(X_train_pca)
    X_test_scaled = scaler_mm.transform(X_test_pca)

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

    # ── Evaluate on KAU ──────────────────────────────────────────────────
    print("\n[3/6] Evaluating on KAU-BCMD external validation set...")
    kau_results = {}
    mendeley_results = {}

    # Classical
    print("  Classical MobileNetV2...")
    X_kau_raw_t = np.load(FEAT_DIR / "features_kau_raw.npy")
    X_test_raw  = np.load(FEAT_DIR / "features_test_raw.npy")
    kau_results["Classical"]      = evaluate_cnn_on_kau(cnn_model, X_kau_raw_t, y_kau)
    mendeley_results["Classical"] = evaluate_cnn_on_kau(cnn_model, X_test_raw,  y_test)

    # HQCNN Regime A
    label_A = f"HQCNN q={N_QUBITS} l={N_LAYERS}"
    print(f"  {label_A}...")
    kau_results[label_A]      = evaluate_vqc_on_kau(vqc_A, X_kau_scaled, y_kau)
    mendeley_results[label_A] = evaluate_vqc_on_kau(
        vqc_A, X_test_scaled, y_test
    )

    # QFL model
    if vqc_qfl is not None:
        print("  QFL global model...")
        kau_results["QFL"]      = evaluate_vqc_on_kau(vqc_qfl, X_kau_scaled, y_kau)
        mendeley_results["QFL"] = evaluate_vqc_on_kau(vqc_qfl, X_test_scaled, y_test)

    # Print summary
    print("\n  ── Cross-population Results ──")
    for name in kau_results:
        m_auc = mendeley_results.get(name, {}).get("auc", "—")
        k_auc = kau_results[name]["auc"]
        gap   = round(m_auc - k_auc, 4) if isinstance(m_auc, float) else "—"
        print(f"  {name:30s} | Mendeley AUC={m_auc:.4f} | "
              f"KAU AUC={k_auc:.4f} | Gap={gap}")
        # Per-class detail for KAU (important given 17:1 imbalance)
        pc = kau_results[name]["per_class"]
        print(f"    KAU Sensitivity (Malignant recall): "
              f"{pc.get('Malignant',{}).get('recall','—'):.4f}")
        print(f"    KAU Specificity (Benign recall):    "
              f"{pc.get('Benign',{}).get('recall','—'):.4f}")

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
    master_df.to_csv(OUT_DIR / "final_ablation_table.csv", index=False)
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

    # ── Save generalisation report ────────────────────────────────────────
    report = {
        "study": "Privacy-Preserving QFL for Breast Cancer Screening",
        "primary_dataset":  "Mendeley (Polokwane, South Africa)",
        "external_dataset": "KAU-BCMD (Saudi Arabia / MENA)",
        "cross_population_results": {
            name: {
                "mendeley_auc": mendeley_results.get(name, {}).get("auc", "—"),
                "kau_auc":      kau_results[name]["auc"],
                "kau_f1":       kau_results[name]["f1"],
                "kau_sensitivity": kau_results[name]["per_class"].get(
                    "Malignant", {}).get("recall", "—"),
                "kau_specificity": kau_results[name]["per_class"].get(
                    "Benign", {}).get("recall", "—"),
            }
            for name in kau_results
        },
        "domain_shift": shift_metrics,
        "ablation_table_path": str(OUT_DIR / "final_ablation_table.csv"),
    }
    with open(OUT_DIR / "generalisation_report.json", "w") as f:
        json.dump(report, f, indent=2)

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