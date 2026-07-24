"""
=============================================================================
2A — CLASSICAL BASELINE: MobileNetV2 FINE-TUNING ON MENDELEY DATASET
=============================================================================
Project : Quantum-Enhanced Hybrid Architectures for Mammographic Breast
          Cancer Classification in African and MENA Populations
Purpose : Train a MobileNetV2 (ImageNet pretrained) binary classifier on the
          Mendeley (Polokwane, South Africa) mammography dataset.
          This provides:
            (1) The classical performance ceiling for comparison against HQCNN
            (2) The pretrained backbone whose penultimate-layer features will
                be fed into the quantum VQC pipeline (2b_feature_pca.py)
Output:   ../ppqfl-breast-cancer-screening/outputs/baseline_outputs/
            mobilenetv2_best.pt       ← best checkpoint (val AUC)
            mobilenetv2_final.pt      ← final epoch checkpoint
            training_history.csv      ← per-epoch metrics
            training_curves.png       ← loss + AUC curves
            confusion_matrix.png      ← test set confusion matrix
            roc_curve.png             ← test set ROC curve
            baseline_results.json     ← summary metrics for reporting
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import (
    MobileNet_V2_Weights, ResNet50_Weights, EfficientNet_B0_Weights
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

from pipeline_utils import seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Dataset paths (auto-detect) ──────────────────────────────────────
def find_mendeley_root() -> Path:
    """Auto-detect the Mendeley dataset root directory.

    Searches relative to this script's location for the dataset.
    """
    script_dir = Path(__file__).resolve().parent
    relative_paths = [
        script_dir / "Breast Cancer Dataset/Breast Cancer Original",
        script_dir.parent / "Breast Cancer Dataset/Breast Cancer Original",
        Path("./Breast Cancer Dataset/Breast Cancer Original"),
        Path("../Breast Cancer Dataset/Breast Cancer Original"),
    ]
    for p in relative_paths:
        if p.exists() and (p / "Benign").exists() and (p / "Malignant").exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Mendeley dataset not found. Searched relative paths:\n"
        f"  " + "\n  ".join(str(p) for p in relative_paths) + "\n"
        f"Ensure the dataset is located at one of these paths with Benign/ and Malignant/ subfolders."
    )

ROOT_MENDELEY = find_mendeley_root()
MENDELEY_BENIGN    = ROOT_MENDELEY / "Benign"
MENDELEY_MALIGNANT = ROOT_MENDELEY / "Malignant"

# ── Load split indices saved by _1_eda.py (for reproducibility) ──────────
PROJECT_ROOT     = Path(__file__).resolve().parent
BASE             = PROJECT_ROOT / "outputs"
SPLIT_INDEX_FILE = BASE / "eda_outputs/mendeley_split_indices.json"

OUT_DIR          = BASE / "baseline_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE SELECTOR — enable exactly ONE by setting BACKBONE here
# Options: "mobilenetv2" | "resnet50" | "efficientnet_b0"
# ══════════════════════════════════════════════════════════════════════════════
BACKBONE = "mobilenetv2"

# ── Transfer learning strategy ───────────────────────────────────────────────
# "progressive" (default): freeze backbone, train head for freeze_epochs,
#                           then unfreeze backbone with lower lr_backbone.
#                           Best accuracy; matches Howard & Ruder (ULMFiT).
# "head_only"            : backbone fully frozen throughout. Fewer params,
#                          faster, but lower performance on small datasets.
FREEZE_STRATEGY = "progressive"

# ── Hyperparameters ──────────────────────────────────────────────────────────
CFG = {
    "image_size":    224,
    "batch_size":    16,
    "num_epochs":    30,
    "lr_head":       1e-3,      # LR for classification head
    "lr_backbone":   1e-4,      # LR for backbone after unfreezing (progressive only)
    "weight_decay":  1e-4,
    "patience":      7,
    "num_workers":   2,
    "random_state":  42,
    "freeze_epochs": 5,         # Epochs to keep backbone frozen (progressive only)
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def collect_paths(benign_dir: Path, malignant_dir: Path):
    records = []
    for label_int, label, directory in [
        (0, "Benign",    benign_dir),
        (1, "Malignant", malignant_dir),
    ]:
        for f in directory.rglob("*"):
            if f.suffix.lower() in SUPPORTED_EXT:
                records.append({"path": str(f), "label": label_int})
    return pd.DataFrame(records)


class MammogramDataset(Dataset):
    """
    PyTorch Dataset for mammography classification.
    Accepts a DataFrame with columns: path, label (0/1).
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


# ── Transforms ──────────────────────────────────────────────────────────────
# ImageNet mean/std — MobileNetV2/ResNet-50 was pretrained with these
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((CFG["image_size"] + 32, CFG["image_size"] + 32)),
    transforms.RandomCrop(CFG["image_size"]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((CFG["image_size"], CFG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL: Pluggable backbone with frozen feature extractor + trainable head
#     Strategy: proper transfer learning — backbone FULLY FROZEN, head trained.
#     This is the fair comparison basis for the HQCNN (also few trainable params).
# ══════════════════════════════════════════════════════════════════════════════

def build_backbone(backbone: str = BACKBONE) -> tuple:
    """
    Build a pretrained backbone with a frozen feature extractor and a
    trainable binary classification head.

    Returns (model, feat_dim, ckpt_name) where:
      model     — nn.Module ready for training
      feat_dim  — penultimate feature dimension (used in 2b_feature_pca.py)
      ckpt_name — filename stem for checkpoint saving

    To switch backbone: change BACKBONE constant above.
    """
    if backbone == "mobilenetv2":
        base     = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        feat_dim = base.classifier[1].in_features   # 1280
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        # Freeze everything except classifier
        for name, param in base.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        return base, feat_dim, "mobilenetv2"

    elif backbone == "efficientnet_b0":
        base     = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = base.classifier[1].in_features   # 1280
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        for name, param in base.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        return base, feat_dim, "efficientnet_b0"

    elif backbone == "resnet50":
        base     = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = base.fc.in_features              # 2048
        base.fc  = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2),
        )
        for name, param in base.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        return base, feat_dim, "resnet50"

    else:
        raise ValueError(f"Unknown backbone: '{backbone}'. "
                         "Choose from: mobilenetv2, efficientnet_b0, resnet50")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for images, labels in tqdm(loader, desc="  Eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    # AUC requires both classes and variance in predictions
    unique_labels = set(all_labels)
    unique_probs = set(np.round(all_probs, 6))  # Round to avoid floating point noise
    if len(unique_labels) < 2:
        auc = 0.0  # No variance in labels
    elif len(unique_probs) < 2:
        auc = 0.5  # No variance in predictions (random)
    else:
        auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc, all_labels, all_preds, all_probs


def get_optimizer(model, cfg, backbone_name: str, strategy: str):
    """
    progressive : two param groups — head at lr_head, backbone at lr_backbone.
    head_only   : only trainable (head) params — backbone stays frozen.
    """
    # Identify head param names per backbone
    head_keywords = {
        "mobilenetv2":    "classifier",
        "efficientnet_b0": "classifier",
        "resnet50":       "fc",
    }
    kw = head_keywords.get(backbone_name, "classifier")

    head_params     = [p for n, p in model.named_parameters() if kw in n]
    backbone_params = [p for n, p in model.named_parameters()
                       if kw not in n and p.requires_grad]

    if strategy == "progressive" and backbone_params:
        return optim.AdamW([
            {"params": head_params,     "lr": cfg["lr_head"]},
            {"params": backbone_params, "lr": cfg["lr_backbone"]},
        ], weight_decay=cfg["weight_decay"])
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(trainable, lr=cfg["lr_head"],
                           weight_decay=cfg["weight_decay"])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLOTTING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MobileNetV2 Training Curves — Mendeley (Polokwane, SA)", fontweight="bold")

    for ax, metric, ylabel in [
        (axes[0], ("train_loss", "val_loss"),   "Loss"),
        (axes[1], ("train_acc",  "val_acc"),    "Accuracy"),
        (axes[2], ("val_auc",    None),         "Val AUC-ROC"),
    ]:
        ax.plot(history["epoch"], history[metric[0]], label=metric[0].replace("_", " ").title(), color="#4C72B0")
        if metric[1]:
            ax.plot(history["epoch"], history[metric[1]], label=metric[1].replace("_", " ").title(),
                    color="#DD8452", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(f"{ylabel} vs Epoch")
        if "best_epoch" in history.columns:
            best_ep = history.loc[history["is_best"] == True, "epoch"]
            for ep in best_ep:
                ax.axvline(ep, color="green", alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Test Set"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {title}\n{BACKBONE} Classical Baseline")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    # plt.show() 


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — MobileNetV2 Classical Baseline\n(Mendeley Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    # plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print(f"  2A — {BACKBONE} CLASSICAL BASELINE")
    print("  QML Breast Cancer Classification | Mendeley Dataset")
    print("═"*70)

    # ── Cache guard ──────────────────────────────────────────────────────────
    from cache_check import already_done, CACHE
    if already_done("baseline"):
        print("  Loading best checkpoint for return value...")
        model, _, ckpt_stem = build_backbone(BACKBONE)
        model = model.to(DEVICE)
        ckpt = OUT_DIR / f"{ckpt_stem}_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model
    # ────────────────────────────────────────────────────────────────────────

    # ── Load all paths ───────────────────────────────────────────────────────
    print("\n[1/6] Loading image paths...")
    df_all = collect_paths(MENDELEY_BENIGN, MENDELEY_MALIGNANT)
    print(f"  Total images: {len(df_all)}")

    # ── Load split indices (generated in 1_eda.py) ───────────────────────
    if SPLIT_INDEX_FILE.exists():
        print(f"[2/6] Loading split indices from {SPLIT_INDEX_FILE}...")
        with open(SPLIT_INDEX_FILE) as f:
            split_info = json.load(f)
        train_df = df_all.iloc[split_info["train_indices"]].reset_index(drop=True)
        val_df   = df_all.iloc[split_info["val_indices"]].reset_index(drop=True)
        test_df  = df_all.iloc[split_info["test_indices"]].reset_index(drop=True)
    else:
        print("[2/6] Split index file not found — computing splits now...")
        print("      (Run 1_eda.py first for full reproducibility.)")
        labels = df_all["label"].values
        train_idx, temp_idx = train_test_split(
            range(len(df_all)), test_size=0.30, stratify=labels, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=labels[list(temp_idx)], random_state=42
        )
        train_df = df_all.iloc[list(train_idx)].reset_index(drop=True)
        val_df   = df_all.iloc[list(val_idx)].reset_index(drop=True)
        test_df  = df_all.iloc[list(test_idx)].reset_index(drop=True)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    # Log class distribution for sanity check
    train_class_dist = train_df["label"].value_counts().to_dict()
    val_class_dist = val_df["label"].value_counts().to_dict()
    test_class_dist = test_df["label"].value_counts().to_dict()
    print(f"  Train class dist: {train_class_dist}")
    print(f"  Val class dist: {val_class_dist}")
    print(f"  Test class dist: {test_class_dist}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("[3/6] Building DataLoaders...")
    train_ds = MammogramDataset(train_df, transform=train_transform)
    val_ds   = MammogramDataset(val_df,   transform=eval_transform)
    test_ds  = MammogramDataset(test_df,  transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=CFG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

    # ── Model, criterion, optimiser ──────────────────────────────────────────
    print("[4/6] Building model...")
    model, feat_dim, ckpt_stem = build_backbone(BACKBONE)
    model = model.to(DEVICE)
    CKPT_BEST  = OUT_DIR / f"{ckpt_stem}_best.pt"
    CKPT_FINAL = OUT_DIR / f"{ckpt_stem}_final.pt"

    # Class weights for imbalance (benign > malignant in Mendeley)
    class_counts = train_df["label"].value_counts().sort_index()
    # Ensure both classes (0 and 1) are present, default to 0 if missing
    class_0_count = class_counts.get(0, 0)
    class_1_count = class_counts.get(1, 0)
    # Avoid division by zero - use a small epsilon if a class has no samples
    class_weights = torch.tensor(
        [1.0 / max(class_0_count, 1), 1.0 / max(class_1_count, 1)], dtype=torch.float
    ).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * 2  # normalise
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = get_optimizer(model, CFG, BACKBONE, FREEZE_STRATEGY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["num_epochs"], eta_min=1e-6
    )

    n_trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    strategy_desc  = (
        f"progressive unfreeze after {CFG['freeze_epochs']} epochs "
        f"(head lr={CFG['lr_head']}, backbone lr={CFG['lr_backbone']})"
        if FREEZE_STRATEGY == "progressive"
        else "head-only (backbone fully frozen throughout)"
    )
    print(f"  Backbone : {BACKBONE} | Total: {n_total_params:,} | "
          f"Trainable now: {n_trainable:,}")
    print(f"  Strategy : {strategy_desc}")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[5/6] Training for up to {CFG['num_epochs']} epochs...")
    best_val_auc = -1.0  # Use -1.0 so first epoch (AUC >= 0) will always be best
    best_epoch = 0
    patience_ctr = 0
    history      = []

    for epoch in range(1, CFG["num_epochs"] + 1):

        # Progressive unfreeze: release backbone after freeze_epochs
        if FREEZE_STRATEGY == "progressive" and epoch == CFG["freeze_epochs"] + 1:
            print(f"\n  → Epoch {epoch}: unfreezing backbone "
                  f"(lr={CFG['lr_backbone']})")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = get_optimizer(model, CFG, BACKBONE, FREEZE_STRATEGY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG["num_epochs"] - epoch, eta_min=1e-6
            )

        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, vl_f1, vl_auc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        is_best = vl_auc > best_val_auc
        if is_best:
            best_val_auc = vl_auc
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_BEST)
            patience_ctr = 0
        else:
            patience_ctr += 1

        history.append({
            "epoch":     epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss":   vl_loss, "val_acc":   vl_acc, "val_f1":   vl_f1,
            "val_auc":    vl_auc,  "is_best":   is_best,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"  Epoch {epoch:3d}/{CFG['num_epochs']} | "
              f"TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.4f} | "
              f"VlLoss: {vl_loss:.4f} VlAcc: {vl_acc:.4f} VlAUC: {vl_auc:.4f}"
              + (" ← BEST" if is_best else ""))

        if patience_ctr >= CFG["patience"]:
            print(f"\n  Early stopping at epoch {epoch} (patience={CFG['patience']} exceeded).")
            break

    torch.save(model.state_dict(), CKPT_FINAL)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUT_DIR / "training_history.csv", index=False)
    plot_training_curves(hist_df)

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set (best checkpoint)...")
    model.load_state_dict(torch.load(CKPT_BEST, map_location=DEVICE))
    ts_loss, ts_acc, ts_f1, ts_auc, y_true, y_pred, y_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n  ──── TEST SET RESULTS ────")
    print(f"  Accuracy : {ts_acc:.4f}")
    print(f"  F1-score : {ts_f1:.4f}")
    print(f"  AUC-ROC  : {ts_auc:.4f}")
    print(f"\n  Full classification report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

    plot_confusion_matrix(y_true, y_pred, title="Mendeley Test Set")
    plot_roc_curve(y_true, y_probs)

    results = {
        "model":            f"{BACKBONE} (ImageNet pretrained)",
        "backbone":         BACKBONE,
        "freeze_strategy":  FREEZE_STRATEGY,
        "dataset":          "Mendeley Mammogram (Polokwane, South Africa)",
        "split":           "70/15/15 stratified",
        "test_accuracy":   round(ts_acc,  4),
        "test_f1":         round(ts_f1,   4),
        "test_auc_roc":    round(ts_auc,  4),
        "best_val_auc":    round(best_val_auc, 4),
        "total_params":    n_total_params,
        "trainable_params": n_trainable,
        "config":          CFG,
    }
    with open(OUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ BASELINE TRAINING COMPLETE.")
    print(f"  Best checkpoint : {CKPT_BEST}")
    print(f"  Results summary : {OUT_DIR / 'baseline_results.json'}")
    print(f"  Next step → 2b_feature_pca.py")
    CACHE.mark_done("baseline")
    return model
 

if __name__ == "__main__":
    model = main()