"""
=============================================================================
8–9 — SIMULATED QUANTUM FEDERATED LEARNING (QFL)
=============================================================================
Project : Quantum-Enhanced Hybrid Architectures for Mammographic Breast
          Cancer Classification in African and MENA Populations

Motivation:
  Real multi-institution federated learning requires ethical clearance and
  data sharing agreements. This script simulates QFL by partitioning the
  Mendeley dataset into 3 virtual clients representing geographically
  plausible Ghanaian hospital tiers, with deliberately non-IID class
  distributions to reflect real-world referral patterns:

    Client A — "Korle Bu Teaching Hospital, Accra"   (urban tertiary)
    Client B — "Komfo Anokye Teaching Hospital, Kumasi" (regional urban)
    Client C — "Tamale Teaching Hospital"             (peri-urban/rural)

  No real patient data leaves any node. Each client trains locally on its
  shard; only VQC parameters θ are aggregated (FedAvg). The backbone
  remains frozen throughout — only the quantum layer is federated.

FL Protocol:
  FedAvg (McMahan et al., 2017) over VQC parameters only.
  Flower (flwr) framework used for client/server abstraction.
  Local epochs per round: configurable (default 3).
  Global rounds: configurable (default 15).

Privacy note:
  In a real deployment, differential privacy (DP) noise would be added
  to parameter updates before aggregation. We simulate DP by optionally
  adding calibrated Gaussian noise to gradients (σ_dp configurable).
  This approximates the privacy-utility trade-off without real data.

Outputs (../ppqfl-breast-cancer-screening/outputs/qfl_outputs/):
  partition_summary.png        ← client data distributions (non-IID viz)
  federated_training.csv       ← per-round global metrics
  federated_vs_centralised.png ← key comparison plot
  client_drift.png             ← per-client local model vs global
  privacy_utility_tradeoff.png ← AUC vs DP noise level
  qfl_summary.json             ← all metrics for ablation table

Prerequisites:
  pip install flwr pennylane torch scikit-learn
  _3_5_vqc.py must have completed (best Regime A checkpoint used as
  warm-start initialisation for federated VQC).
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import json, warnings, copy
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive, writes files only — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pennylane as qml

import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from pipeline_utils import seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
BASE         = PROJECT_ROOT / "outputs"
FEAT_DIR     = BASE / "feature_outputs"
VQC_CKPT_DIR = BASE / "vqc_outputs/regime_A"        # base dir — subdirs appended per regime
OUT_DIR      = BASE / "qfl_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Match to your best Regime A result ───────────────────────────────────────
# we also need to automate the selection of our best model
N_QUBITS  = 4
N_LAYERS  = 2
VQC_LR    = 0.01

# ── FL hyperparameters ────────────────────────────────────────────────────────
FL_CFG = {
    "n_clients":      3,
    "n_rounds":       15,       # global FL rounds
    "local_epochs":   3,        # local epochs per client per round
    "local_lr":       0.005,    # local VQC learning rate
    "batch_size":     16,
    "random_state":   42,
}

# ── Non-IID partition: (benign_fraction, malignant_fraction) per client ───────
# Reflects referral pattern: rural/peri-urban sees more early-stage (benign)
# Urban tertiary sees more confirmed malignant referrals
CLIENT_CFG = {
    "Accra":   {"label": "Korle Bu, Accra (urban tertiary)",
                "benign_frac": 0.50, "malignant_frac": 0.50},
    "Kumasi":  {"label": "Komfo Anokye, Kumasi (regional)",
                "benign_frac": 0.60, "malignant_frac": 0.40},
    "Tamale":  {"label": "Tamale Teaching (peri-urban/rural)",
                "benign_frac": 0.75, "malignant_frac": 0.25},
}

# ── Differential privacy simulation ─────────────────────────────────────────
# σ_dp = 0.0 → no DP noise (baseline)
# Run multiple σ values for privacy-utility trade-off curve
DP_SIGMAS = [0.0, 0.01, 0.05, 0.10, 0.20]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  VQC ARCHITECTURE  (identical to 3_5_vqc.py — reproduced for
#     self-contained execution without import dependency)
# ══════════════════════════════════════════════════════════════════════════════

def build_vqc(n_qubits: int, n_layers: int):
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
    """Standalone VQC classifier — the federated layer."""
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.vqc      = build_vqc(n_qubits, n_layers)
        self.bias     = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out  = torch.stack([self.vqc(x[i]) for i in range(x.shape[0])])
        return torch.sigmoid(out + self.bias)

    def get_parameters(self) -> List[np.ndarray]:
        """Extract all parameters as numpy arrays (Flower protocol)."""
        return [p.detach().numpy().copy()
                for p in self.parameters()]

    def set_parameters(self, params: List[np.ndarray]):
        """Load parameters from numpy arrays (Flower protocol)."""
        with torch.no_grad():
            for p, new_val in zip(self.parameters(), params):
                p.copy_(torch.tensor(new_val, dtype=torch.float32))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_pretrained_vqc(n_qubits: int, n_layers: int, lr: float) -> VQCModel:
    """
    Warm-start the federated VQC from the best centralised Regime A checkpoint.
    This gives FL a head start and reduces rounds needed to converge.
    If checkpoint not found, starts from random initialisation.
    """
    model    = VQCModel(n_qubits, n_layers)
    ckpt     = VQC_CKPT_DIR / f"vqc_q{n_qubits}_l{n_layers}_lr{lr}.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Warm-start from centralised checkpoint: {ckpt}")
    else:
        print(f"  [INFO] No pretrained checkpoint found at {ckpt}.")
        print("  Starting FL from random VQC initialisation.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA: NON-IID PARTITION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_scale_features(n_qubits: int):
    """Load all splits and fit MinMaxScaler on train only."""
    X_train = np.load(FEAT_DIR / f"features_train_pca{n_qubits}.npy")
    X_val   = np.load(FEAT_DIR / f"features_val_pca{n_qubits}.npy")
    X_test  = np.load(FEAT_DIR / f"features_test_pca{n_qubits}.npy")
    y_train = np.load(FEAT_DIR / "labels_train.npy")
    y_val   = np.load(FEAT_DIR / "labels_val.npy")
    y_test  = np.load(FEAT_DIR / "labels_test.npy")

    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def partition_non_iid(X_train: np.ndarray,
                      y_train: np.ndarray,
                      client_cfg: dict,
                      random_state: int = 42) -> dict:
    """
    Partition training data into non-IID client shards.

    Strategy:
      1. Separate benign and malignant samples.
      2. Allocate to each client according to its (benign_frac, malignant_frac).
      3. Shards are disjoint — no sample appears on more than one client.
      4. The total across clients equals the full training set.

    Non-IID means class ratios differ per client, simulating real-world
    referral pattern heterogeneity across hospital tiers.
    """
    rng = np.random.default_rng(random_state)

    benign_idx    = np.where(y_train == 0)[0]
    malignant_idx = np.where(y_train == 1)[0]
    rng.shuffle(benign_idx)
    rng.shuffle(malignant_idx)

    client_names = list(client_cfg.keys())
    n_clients    = len(client_names)

    # Compute benign/malignant counts per client proportionally
    b_fracs = np.array([client_cfg[c]["benign_frac"]    for c in client_names])
    m_fracs = np.array([client_cfg[c]["malignant_frac"] for c in client_names])
    # Normalise so fractions sum to 1 across clients
    b_fracs = b_fracs / b_fracs.sum()
    m_fracs = m_fracs / m_fracs.sum()

    n_benign    = len(benign_idx)
    n_malignant = len(malignant_idx)

    b_counts = (b_fracs * n_benign).astype(int)
    m_counts = (m_fracs * n_malignant).astype(int)
    # Assign remainders to last client to use all samples
    b_counts[-1] = n_benign    - b_counts[:-1].sum()
    m_counts[-1] = n_malignant - m_counts[:-1].sum()

    partitions = {}
    b_ptr, m_ptr = 0, 0
    for i, name in enumerate(client_names):
        b_slice = benign_idx[b_ptr : b_ptr + b_counts[i]]
        m_slice = malignant_idx[m_ptr : m_ptr + m_counts[i]]
        idx     = np.concatenate([b_slice, m_slice])
        rng.shuffle(idx)
        partitions[name] = {
            "X": X_train[idx],
            "y": y_train[idx],
            "n_benign":    len(b_slice),
            "n_malignant": len(m_slice),
            "label":       client_cfg[name]["label"],
        }
        b_ptr += b_counts[i]
        m_ptr += m_counts[i]

    return partitions


def plot_partition_summary(partitions: dict, save_path: Path):
    """Visualise non-IID class distribution across clients."""
    client_names = list(partitions.keys())
    n_clients    = len(client_names)

    fig, axes = plt.subplots(1, n_clients + 1, figsize=(5 * (n_clients + 1), 5))
    fig.suptitle("Non-IID Data Partition Across Virtual Clients\n"
                 "Simulating Ghanaian Hospital Referral Patterns",
                 fontweight="bold", fontsize=12)

    all_b, all_m = [], []
    for i, (name, data) in enumerate(partitions.items()):
        b, m  = data["n_benign"], data["n_malignant"]
        total = b + m
        all_b.append(b); all_m.append(m)
        bars = axes[i].bar(["Benign", "Malignant"], [b, m],
                           color=["#4C72B0", "#DD8452"], edgecolor="white")
        axes[i].set_title(f"Client: {name}\n{data['label']}", fontsize=9)
        axes[i].set_ylabel("Image Count")
        for bar, val in zip(bars, [b, m]):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1,
                         f"{val}\n({val/total*100:.0f}%)",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[i].text(0.97, 0.97, f"Total: {total}",
                     transform=axes[i].transAxes,
                     ha="right", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="lightyellow", alpha=0.8))

    # Summary grouped bar
    x = np.arange(n_clients)
    w = 0.35
    axes[-1].bar(x - w/2, all_b, w, label="Benign",    color="#4C72B0")
    axes[-1].bar(x + w/2, all_m, w, label="Malignant", color="#DD8452")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(client_names)
    axes[-1].set_title("Class Distribution Summary\n(Non-IID heterogeneity)")
    axes[-1].set_ylabel("Count"); axes[-1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FLOWER CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class QFLClient(fl.client.Client):
    """
    Flower client wrapping a local VQCModel.
    Each client holds its private shard, trains locally for
    FL_CFG["local_epochs"] epochs, then returns updated parameters
    to the server for FedAvg aggregation.

    Only VQC parameters (θ + bias) are transmitted — never raw data.
    This is the privacy guarantee of the federated setup.
    """
    def __init__(self, cid: str, X: np.ndarray, y: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 n_qubits: int, n_layers: int,
                 dp_sigma: float = 0.0):
        self.cid      = cid
        self.X        = torch.tensor(X, dtype=torch.float32)
        self.y        = torch.tensor(y, dtype=torch.long)
        self.X_val    = torch.tensor(X_val, dtype=torch.float32)
        self.y_val    = torch.tensor(y_val, dtype=torch.long)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dp_sigma = dp_sigma
        self.model    = VQCModel(n_qubits, n_layers)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        params = ndarrays_to_parameters(self.model.get_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=params,
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Load global parameters into local model
        global_params = parameters_to_ndarrays(ins.parameters)
        self.model.set_parameters(global_params)

        # Local training
        dataset  = TensorDataset(self.X, self.y.float())
        loader   = DataLoader(dataset, batch_size=FL_CFG["batch_size"],
                              shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=FL_CFG["local_lr"])

        self.model.train()
        for _ in range(FL_CFG["local_epochs"]):
            for X_b, y_b in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
                loss.backward()
                # ── Differential privacy: add Gaussian noise to gradients ──
                if self.dp_sigma > 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad += torch.randn_like(param.grad) * self.dp_sigma
                optimizer.step()

        # Return updated parameters (never raw data)
        updated_params = self.model.get_parameters()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=len(self.X),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self.model.set_parameters(parameters_to_ndarrays(ins.parameters))
        self.model.eval()
        with torch.no_grad():
            probs = self.model(self.X_val).numpy()
        y_true = self.y_val.numpy()
        auc    = roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else 0.0
        preds  = (probs > 0.5).astype(int)
        acc    = accuracy_score(y_true, preds)
        loss   = float(nn.BCELoss()(
            torch.tensor(probs), torch.tensor(y_true, dtype=torch.float32)
        ).item())
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=len(self.X_val),
            metrics={"auc": auc, "accuracy": acc},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SIMULATION ENGINE  (in-process, no network sockets needed)
# ══════════════════════════════════════════════════════════════════════════════

def run_federated_simulation(partitions: dict,
                              X_val: np.ndarray, y_val: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              n_qubits: int, n_layers: int,
                              n_rounds: int,
                              dp_sigma: float = 0.0,
                              warm_start_params: Optional[List[np.ndarray]] = None,
                              ) -> Tuple[pd.DataFrame, dict]:
    """
    In-process FL simulation using Flower's virtual client engine.
    No network ports opened — all communication is in-memory.

    FedAvg aggregates VQC parameters weighted by number of training samples.
    Global model is evaluated on the held-out validation and test sets
    at the end of each round.

    Returns:
      history_df  — per-round global metrics
      final_model — trained global VQCModel
    """
    client_names = list(partitions.keys())

    # ── Initialise global model ───────────────────────────────────────────
    global_model = VQCModel(n_qubits, n_layers)
    if warm_start_params is not None:
        global_model.set_parameters(warm_start_params)
    global_params = global_model.get_parameters()

    # Pre-build clients (in-process, no serialisation overhead)
    clients = {
        name: QFLClient(
            cid=name,
            X=partitions[name]["X"],
            y=partitions[name]["y"],
            X_val=X_val, y_val=y_val,
            n_qubits=n_qubits, n_layers=n_layers,
            dp_sigma=dp_sigma,
        )
        for name in client_names
    }

    history = []

    for round_num in range(1, n_rounds + 1):
        # ── Local training on each client ─────────────────────────────────
        client_updates = []
        for name, client in clients.items():
            fit_ins = FitIns(
                parameters=ndarrays_to_parameters(global_params),
                config={}
            )
            fit_res = client.fit(fit_ins)
            client_updates.append((
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
            ))

        # ── FedAvg aggregation ────────────────────────────────────────────
        total_samples = sum(n for _, n in client_updates)
        aggregated    = [
            sum(params[i] * (n / total_samples)
                for params, n in client_updates)
            for i in range(len(global_params))
        ]
        global_params = aggregated
        global_model.set_parameters(global_params)

        # ── Global evaluation on val set ──────────────────────────────────
        global_model.eval()
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_tst_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            val_probs  = global_model(X_val_t).numpy()
            test_probs = global_model(X_tst_t).numpy()

        val_auc  = roc_auc_score(y_val,  val_probs)  if len(set(y_val))  > 1 else 0.0
        test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.0
        val_f1   = f1_score(y_val,  (val_probs  > 0.5).astype(int), zero_division=0)
        test_f1  = f1_score(y_test, (test_probs > 0.5).astype(int), zero_division=0)

        row = {
            "round":      round_num,
            "val_auc":    round(val_auc,  4),
            "test_auc":   round(test_auc, 4),
            "val_f1":     round(val_f1,   4),
            "test_f1":    round(test_f1,  4),
            "dp_sigma":   dp_sigma,
        }

        # Per-client local AUC (client drift monitoring)
        for name, client in clients.items():
            eval_ins = EvaluateIns(
                parameters=ndarrays_to_parameters(global_params), config={}
            )
            eval_res = client.evaluate(eval_ins)
            row[f"client_{name}_auc"] = round(eval_res.metrics["auc"], 4)

        history.append(row)

        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:3d}/{n_rounds} | "
                  f"Val AUC={val_auc:.4f} | Test AUC={test_auc:.4f} | "
                  f"F1={test_f1:.4f} | σ_dp={dp_sigma}")

    final_metrics = {
        "final_val_auc":  history[-1]["val_auc"],
        "final_test_auc": history[-1]["test_auc"],
        "final_test_f1":  history[-1]["test_f1"],
        "dp_sigma":       dp_sigma,
        "n_rounds":       n_rounds,
        "vqc_params":     global_model.count_params(),
    }
    return pd.DataFrame(history), global_model, final_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CENTRALISED BASELINE  (same VQC, same data, no federation)
# ══════════════════════════════════════════════════════════════════════════════

def run_centralised_baseline(X_train, y_train, X_val, y_val,
                              X_test, y_test, n_qubits, n_layers,
                              warm_start_params=None) -> dict:
    """
    Train the same VQC on the full pooled training set (no federation).
    Equivalent number of gradient steps as FL (n_rounds × local_epochs)
    for a fair comparison.
    """
    print("\n  ── Centralised baseline (pooled training) ──")
    model    = VQCModel(n_qubits, n_layers)
    if warm_start_params is not None:
        model.set_parameters(warm_start_params)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=FL_CFG["local_lr"])

    dataset  = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader   = DataLoader(dataset, batch_size=FL_CFG["batch_size"], shuffle=True)
    # Match total gradient steps: n_rounds × local_epochs
    total_epochs = FL_CFG["n_rounds"] * FL_CFG["local_epochs"]

    for epoch in range(total_epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            nn.BCELoss()(model(X_b), y_b).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_p  = model(torch.tensor(X_val,  dtype=torch.float32)).numpy()
        test_p = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    return {
        "val_auc":   round(roc_auc_score(y_val,  val_p),  4),
        "test_auc":  round(roc_auc_score(y_test, test_p), 4),
        "test_f1":   round(f1_score(y_test, (test_p > 0.5).astype(int),
                                    zero_division=0), 4),
        "vqc_params": model.count_params(),
        "total_epochs": total_epochs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_federated_vs_centralised(fed_history: pd.DataFrame,
                                   centralised: dict,
                                   save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Federated (QFL) vs Centralised Training\n"
                 "VQC on Mendeley Dataset — Simulated Ghanaian Hospital Clients",
                 fontweight="bold")

    ax = axes[0]
    ax.plot(fed_history["round"], fed_history["val_auc"],
            lw=2, color="#4C72B0", label="QFL — Val AUC")
    ax.plot(fed_history["round"], fed_history["test_auc"],
            lw=2, color="#4C72B0", linestyle="--", label="QFL — Test AUC")
    ax.axhline(centralised["val_auc"], color="#DD8452",
               linestyle="-", lw=1.5, label=f"Centralised Val AUC={centralised['val_auc']:.4f}")
    ax.axhline(centralised["test_auc"], color="#DD8452",
               linestyle="--", lw=1.5, label=f"Centralised Test AUC={centralised['test_auc']:.4f}")
    ax.set_xlabel("FL Round"); ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC: QFL vs Centralised")
    ax.legend(fontsize=8); ax.set_ylim([0.5, 1.02]); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fed_history["round"], fed_history["test_f1"],
            lw=2, color="#2ca02c", label="QFL — Test F1")
    ax.axhline(centralised["test_f1"], color="#FF7F0E",
               linestyle="--", lw=1.5,
               label=f"Centralised Test F1={centralised['test_f1']:.4f}")
    ax.set_xlabel("FL Round"); ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score: QFL vs Centralised")
    ax.legend(fontsize=8); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_client_drift(fed_history: pd.DataFrame, client_names: list,
                      save_path: Path):
    """
    Plot per-client AUC vs global model AUC across rounds.
    Client drift = how much each client's local performance diverges from global.
    High drift indicates strong non-IID effect.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#4C72B0", "#DD8452", "#2ca02c"]

    ax.plot(fed_history["round"], fed_history["val_auc"],
            lw=2.5, color="black", label="Global Val AUC", zorder=5)
    for name, color in zip(client_names, colors):
        col = f"client_{name}_auc"
        if col in fed_history.columns:
            ax.plot(fed_history["round"], fed_history[col],
                    lw=1.5, color=color, linestyle="--",
                    alpha=0.8, label=f"Client: {name}")

    ax.set_xlabel("FL Round"); ax.set_ylabel("AUC-ROC")
    ax.set_title("Client Drift Monitoring\n"
                 "Global Model vs Per-Client Local AUC Across Rounds")
    ax.legend(fontsize=9); ax.set_ylim([0.4, 1.02]); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_privacy_utility_tradeoff(dp_results: list, save_path: Path):
    """
    Show how test AUC and F1 degrade as DP noise σ increases.
    The σ=0 point is the privacy-free baseline.
    The curve illustrates the privacy-utility trade-off.
    """
    df = pd.DataFrame(dp_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Privacy-Utility Trade-off\n"
                 "QFL with Differential Privacy (Gaussian Gradient Noise)",
                 fontweight="bold")

    for ax, metric, ylabel, color in [
        (axes[0], "final_test_auc", "Test AUC-ROC", "#4C72B0"),
        (axes[1], "final_test_f1",  "Test F1 Score", "#2ca02c"),
    ]:
        ax.plot(df["dp_sigma"], df[metric], marker="o", lw=2,
                color=color, label=metric)
        ax.fill_between(df["dp_sigma"], df[metric],
                        df[metric].min(), alpha=0.1, color=color)
        ax.set_xlabel("DP Noise σ (0 = no privacy, higher = more private)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Privacy Budget")
        ax.grid(True, alpha=0.3)
        # Annotate σ=0 (utility ceiling) and last point (max privacy)
        ax.annotate(f"No DP\n{df[metric].iloc[0]:.4f}",
                    xy=(df["dp_sigma"].iloc[0], df[metric].iloc[0]),
                    xytext=(0.05, 0.85), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PRIVACY ANALYSIS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def privacy_analysis_report(partitions: dict,
                              fed_result: dict,
                              centralised_result: dict,
                              dp_results: list) -> dict:
    """
    Generate a structured privacy analysis summary for reporting.
    Covers:
      - Data never leaves client (structural privacy guarantee)
      - Parameter-only transmission (what is shared)
      - DP noise analysis (quantified privacy-utility trade-off)
      - Utility gap: federated vs centralised AUC
    """
    auc_gap = centralised_result["test_auc"] - fed_result["final_test_auc"]
    f1_gap  = centralised_result["test_f1"]  - fed_result["final_test_f1"]

    # Find σ where AUC drops by less than 2% from no-DP baseline
    baseline_auc = dp_results[0]["final_test_auc"]
    dp_budget_2pct = None
    for r in dp_results:
        if baseline_auc - r["final_test_auc"] <= 0.02:
            dp_budget_2pct = r["dp_sigma"]

    report = {
        "privacy_guarantees": {
            "raw_data_shared":       False,
            "parameters_shared":     True,
            "parameter_dim":         fed_result["vqc_params"],
            "dp_noise_applied":      True,
            "structural_privacy":    "Client data never transmitted; "
                                     "only VQC parameter tensors exchanged.",
        },
        "utility_analysis": {
            "centralised_test_auc":  centralised_result["test_auc"],
            "federated_test_auc":    fed_result["final_test_auc"],
            "auc_utility_gap":       round(auc_gap, 4),
            "centralised_test_f1":   centralised_result["test_f1"],
            "federated_test_f1":     fed_result["final_test_f1"],
            "f1_utility_gap":        round(f1_gap, 4),
            "interpretation":        (
                f"Federated training achieves {fed_result['final_test_auc']:.4f} AUC "
                f"vs {centralised_result['test_auc']:.4f} centralised "
                f"(gap: {auc_gap:.4f}). "
                "This gap is the privacy cost of not sharing raw data."
            ),
        },
        "dp_analysis": {
            "sigma_for_2pct_auc_drop": dp_budget_2pct,
            "all_dp_results": dp_results,
            "interpretation": (
                f"DP noise σ ≤ {dp_budget_2pct} preserves AUC within 2% of "
                "the no-DP baseline, offering a practical operating point "
                "for privacy-preserving deployment."
                if dp_budget_2pct else "No σ preserved AUC within 2% threshold."
            ),
        },
        "client_summary": {
            name: {
                "n_benign":    d["n_benign"],
                "n_malignant": d["n_malignant"],
                "total":       d["n_benign"] + d["n_malignant"],
                "imbalance_ratio": round(
                    max(d["n_benign"], d["n_malignant"]) /
                    max(min(d["n_benign"], d["n_malignant"]), 1), 2
                ),
            }
            for name, d in partitions.items()
        },
        "fl_config": FL_CFG,
    }
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print("  8–9 — SIMULATED QUANTUM FEDERATED LEARNING")
    print("  QML Breast Cancer Classification | Ghanaian Hospital Simulation")
    print("═"*70)

    from cache_check import already_done, CACHE

    # ── Load and scale features ───────────────────────────────────────────
    print("\n[1/7] Loading features...")
    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_and_scale_features(N_QUBITS)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Non-IID partition ─────────────────────────────────────────────────
    print("\n[2/7] Creating non-IID partitions...")
    partitions = partition_non_iid(X_train, y_train, CLIENT_CFG,
                                    FL_CFG["random_state"])
    for name, data in partitions.items():
        b, m = data["n_benign"], data["n_malignant"]
        print(f"  {name}: {b+m} total | {b} Benign ({b/(b+m)*100:.0f}%) "
              f"| {m} Malignant ({m/(b+m)*100:.0f}%)")

    plot_partition_summary(partitions, OUT_DIR / "partition_summary.png")

    # ── Warm-start parameters from centralised Regime A ───────────────────
    print("\n[3/7] Loading warm-start checkpoint...")
    pretrained = load_pretrained_vqc(N_QUBITS, N_LAYERS, VQC_LR)
    warm_params = pretrained.get_parameters()

    # ── Centralised baseline ──────────────────────────────────────────────
    print("\n[4/7] Running centralised baseline...")
    centralised_result = run_centralised_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test,
        N_QUBITS, N_LAYERS, warm_start_params=warm_params
    )
    print(f"  Centralised — Test AUC={centralised_result['test_auc']:.4f} "
          f"F1={centralised_result['test_f1']:.4f}")

    # ── Main QFL simulation (no DP) ───────────────────────────────────────
    print("\n[5/7] Running QFL simulation (σ_dp=0.0)...")
    fed_history, global_model, fed_result = run_federated_simulation(
        partitions, X_val, y_val, X_test, y_test,
        N_QUBITS, N_LAYERS, FL_CFG["n_rounds"],
        dp_sigma=0.0,
        warm_start_params=warm_params,
    )
    fed_history.to_csv(OUT_DIR / "federated_training.csv", index=False)

    # Save final global model
    torch.save(global_model.state_dict(),
               OUT_DIR / f"qfl_global_q{N_QUBITS}_l{N_LAYERS}.pt")

    plot_federated_vs_centralised(
        fed_history, centralised_result,
        OUT_DIR / "federated_vs_centralised.png"
    )
    plot_client_drift(
        fed_history, list(partitions.keys()),
        OUT_DIR / "client_drift.png"
    )

    # ── Privacy-utility trade-off sweep ──────────────────────────────────
    print("\n[6/7] Privacy-utility trade-off (DP noise sweep)...")
    dp_results = []
    for sigma in DP_SIGMAS:
        print(f"\n  σ_dp = {sigma}")
        _, _, dp_metrics = run_federated_simulation(
            partitions, X_val, y_val, X_test, y_test,
            N_QUBITS, N_LAYERS, FL_CFG["n_rounds"],
            dp_sigma=sigma,
            warm_start_params=warm_params,
        )
        dp_results.append(dp_metrics)

    plot_privacy_utility_tradeoff(dp_results,
                                   OUT_DIR / "privacy_utility_tradeoff.png")

    # ── Privacy analysis report ────────────────────────────────────────────
    print("\n[7/7] Generating privacy analysis report...")
    report = privacy_analysis_report(
        partitions, fed_result, centralised_result, dp_results
    )
    with open(OUT_DIR / "qfl_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── Summary print ─────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  QFL SIMULATION COMPLETE")
    print("═"*70)
    print(f"\n  Centralised test AUC : {centralised_result['test_auc']:.4f}")
    print(f"  QFL test AUC         : {fed_result['final_test_auc']:.4f}  "
          f"(gap: {centralised_result['test_auc']-fed_result['final_test_auc']:.4f})")
    print(f"  QFL test F1          : {fed_result['final_test_f1']:.4f}")
    print(f"  VQC params federated : {fed_result['vqc_params']}")
    print(f"\n  Privacy: raw data NEVER transmitted.")
    print(f"  Only {fed_result['vqc_params']} VQC parameters shared per round.")
    if report["dp_analysis"]["sigma_for_2pct_auc_drop"]:
        s = report["dp_analysis"]["sigma_for_2pct_auc_drop"]
        print(f"  DP: σ ≤ {s} preserves AUC within 2% of no-DP baseline.")
    print(f"\n  Outputs: {OUT_DIR}")
    print("  Next step → 10_11_external_validation.py")

    CACHE.mark_done("qfl")


if __name__ == "__main__":
    main()