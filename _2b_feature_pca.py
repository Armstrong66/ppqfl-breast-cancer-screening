"""
=============================================================================
2B — FEATURE EXTRACTION + PCA: BRIDGE TO THE QUANTUM PIPELINE
=============================================================================
Project : Quantum-Enhanced Hybrid Architectures for Mammographic Breast
          Cancer Classification in African and MENA Populations
Purpose : Extract penultimate-layer features from the trained MobileNetV2
          backbone, apply PCA to reduce dimensionality to the quantum-
          feasible range (4–8 components), and validate that the compressed
          features retain class separability.
          This is the direct input to the VQC in Phase 3 (3–5).

Pipeline:
  MobileNetV2 (frozen, best checkpoint)
    → 2048-dim penultimate feature vector (avgpool output)
    → PCA (fit on TRAIN only — no data leakage)
    → 4 / 6 / 8 principal components
    → Saved as .npy arrays ready for quantum encoding

Environment:
Outputs (../ppqfl-breast-cancer-screening/outputs/feature_outputs/):
  features_train_raw.npy   ← 2048-dim raw features, training split
  features_val_raw.npy
  features_test_raw.npy
  features_kau_raw.npy     ← KAU-BCMD external validation features
  labels_train.npy
  labels_val.npy
  labels_test.npy
  labels_kau.npy
  pca_{n}_components.pkl   ← fitted PCA objects (4, 6, 8 components)
  features_train_pca{n}.npy
  features_val_pca{n}.npy
  features_test_pca{n}.npy
  features_kau_pca{n}.npy
  pca_analysis.png         ← explained variance + 2D visualisation
  feature_separability.png ← t-SNE of raw vs PCA features by class
  pca_report.json          ← explained variance ratios for reporting
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os, json, pickle, warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from pipeline_utils import seed_everything

warnings.filterwarnings("ignore")
seed_everything(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Dataset paths (auto-detect) ──────────────────────────────────────
def find_mendeley_root() -> Path:
    """Auto-detect the Mendeley dataset root directory.

    Searches relative to this script's location first, then checks
    common absolute paths used on server environments.
    """
    script_dir = Path(__file__).resolve().parent
    relative_paths = [
        script_dir / "Breast Cancer Dataset/Breast Cancer Original",
        script_dir.parent / "Breast Cancer Dataset/Breast Cancer Original",
        Path("./Breast Cancer Dataset/Breast Cancer Original"),
        Path("../Breast Cancer Dataset/Breast Cancer Original"),
    ]
    # Server paths as fallback
    server_paths = [
        Path("/data/derrick/mendeley/Breast Cancer Dataset/Breast Cancer Original"),
        Path("/data/derrick/mendeley/Breast Cancer Original"),
    ]
    for p in relative_paths + server_paths:
        if p.exists() and (p / "Benign").exists() and (p / "Malignant").exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Mendeley dataset not found. Searched:\n"
        f"  Relative paths:\n    " + "\n    ".join(str(p) for p in relative_paths) + "\n"
        f"  Server paths:\n    " + "\n    ".join(str(p) for p in server_paths)
    )

def find_kau_root() -> Path:
    """Auto-detect the KAU-BCMD dataset root directory.

    Searches relative to this script's location first, then checks
    common absolute paths used on server environments.
    """
    script_dir = Path(__file__).resolve().parent
    relative_paths = [
        script_dir / "kau",
        script_dir.parent / "kau",
        Path("./kau"),
        Path("../kau"),
    ]
    # Server paths as fallback
    server_paths = [
        Path("/data/derrick/kau"),
        Path("/data/derrick"),
    ]
    for p in relative_paths + server_paths:
        if p.exists() and (
            (p / "BIRAD1").exists() or (p / "b1").exists() or
            (p / "BIRADS_1").exists() or (p / "Metadata.csv").exists() or
            (p / "metadata.csv").exists() or (p / "DICOM Images").exists() or
            (p / "Birad3").exists()
        ):
            return p.resolve()
    for p in relative_paths + server_paths:
        if p.exists() and p.is_dir() and "kau" in p.name.lower():
            return p.resolve()
    raise FileNotFoundError(
        f"KAU-BCMD dataset not found. Searched:\n"
        f"  Relative paths:\n    " + "\n    ".join(str(p) for p in relative_paths) + "\n"
        f"  Server paths:\n    " + "\n    ".join(str(p) for p in server_paths)
    )

def find_kau_metadata(kau_root: Path) -> Optional[Path]:
    """Search for official KAU Metadata.csv in kau_root and adjacent directories."""
    candidate_paths = [
        kau_root / "Metadata.csv",
        kau_root / "metadata.csv",
        kau_root.parent / "Metadata.csv",
        kau_root.parent / "metadata.csv",
        Path("/data/derrick/kau/Metadata.csv"),
        Path("/data/derrick/Metadata.csv"),
        Path("./kau/Metadata.csv"),
        Path("../kau/Metadata.csv"),
    ]
    for cp in candidate_paths:
        if cp.exists() and cp.is_file():
            return cp.resolve()
    for cp in kau_root.glob("*metadata*.csv"):
        return cp.resolve()
    for cp in kau_root.parent.glob("*metadata*.csv"):
        return cp.resolve()
    return None

ROOT_MENDELEY = find_mendeley_root()
MENDELEY_BENIGN    = ROOT_MENDELEY / "Benign"
MENDELEY_MALIGNANT = ROOT_MENDELEY / "Malignant"

ROOT_KAU = find_kau_root()
KAU_BIRAD_MAP = {
    0: [ROOT_KAU / "BIRAD1" / "b1",
        ROOT_KAU / "Birad3" / "b3"],
    1: [ROOT_KAU / "Birad4" / "b4",
        ROOT_KAU / "Birad5" / "Birad5"],
}
KAU_BENIGN    = KAU_BIRAD_MAP[0][0]
KAU_MALIGNANT = KAU_BIRAD_MAP[1][0]

PROJECT_ROOT  = Path(__file__).resolve().parent
BASE          = PROJECT_ROOT / "outputs"

# ── Match this to BACKBONE in _2a_baseline.py ──────────────────────────────
BACKBONE         = "mobilenetv2"   # "mobilenetv2" | "resnet50" | "efficientnet_b0"
BEST_CHECKPOINT  = BASE / "baseline_outputs" / f"{BACKBONE}_best.pt"
SPLIT_INDEX_FILE = BASE / "eda_outputs/mendeley_split_indices.json"

# ── Output ───────────────────────────────────────────────────────────────────
OUT_DIR          = BASE / "feature_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PCA components to evaluate — the quantum pipeline will use these ─────────
# 4 qubits → 4 components (baseline, lowest qubit budget)
# 6 qubits → 6 components
# 8 qubits → 8 components
PCA_N_COMPONENTS = [4, 6, 8]

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 32
NUM_WORKERS  = 2
IMAGE_SIZE   = 224
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

print(f"Device: {DEVICE}")


# ── Mask detection hints (same as _1_eda.py) ─────────────────────────────────
MASK_FOLDER_HINTS = {
    "mask", "masks", "segmentation", "segmentations",
    "ground_truth", "groundtruth", "gt",
}
MASK_FILENAME_HINTS = {
    "mask", "segmentation", "segmentations",
    "ground_truth", "groundtruth", "gt",
}


def is_mask_path(path: Path) -> bool:
    """Detect mask/segmentation derivative files by folder structure or filename hints."""
    parts = [part.lower() for part in path.parts[:-1]]
    if any(any(hint in part for hint in MASK_FOLDER_HINTS) for part in parts):
        return True
    stem = path.stem.lower()
    return any(hint in stem for hint in MASK_FILENAME_HINTS)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def collect_paths(benign_dir: Path, malignant_dir: Path) -> pd.DataFrame:
    """Collect image paths excluding mask/segmentation files."""
    from PIL import Image
    records = []
    skipped_mask_folder = 0
    skipped_binary_mode = 0
    for label_int, directory in [(0, benign_dir), (1, malignant_dir)]:
        files = sorted([f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXT], key=lambda p: str(p))
        for f in files:
            if is_mask_path(f):
                skipped_mask_folder += 1
                continue
            try:
                with Image.open(f) as img:
                    if img.mode == "1":
                        skipped_binary_mode += 1
                        continue
            except Exception:
                pass  # Keep corrupt files for audit later
            records.append({"path": str(f), "label": label_int})
    if skipped_mask_folder > 0:
        print(f"  Skipped {skipped_mask_folder} mask/segmentation files")
    if skipped_binary_mode > 0:
        print(f"  Skipped {skipped_binary_mode} binary mode mask files")
    return pd.DataFrame(records)


def is_kau_mask_path(path: Path) -> bool:
    hints = {"tumor masks", "tumor mask", "mask", "masks", "segmentation",
             "segmentations", "report", "reports", "ground_truth", "groundtruth", "gt"}
    parts = [part.lower() for part in path.parts[:-1]]
    stem = path.stem.lower()
    if any(any(h in part for h in hints) for part in parts):
        return True
    return any(h in stem for h in hints)


def assert_no_format_confound(df_kau: pd.DataFrame, max_allowed_diff: float = 0.25):
    """Verify after filtering that no resolution cluster has an extreme class deviation."""
    if df_kau.empty:
        raise ValueError("Cannot run assert_no_format_confound on empty dataframe.")

    label_col = "label" if "label" in df_kau.columns else "label_int"
    overall_balance = df_kau[label_col].mean()
    n_benign = (df_kau[label_col] == 0).sum()
    n_malignant = (df_kau[label_col] == 1).sum()
    print(f"  [Integrity Audit] KAU-BCMD Cohort: {len(df_kau)} total images "
          f"({n_benign} Benign, {n_malignant} Malignant | balance={overall_balance:.3f})")

    confounds = []
    if "width" in df_kau.columns and "height" in df_kau.columns:
        for (w, h), group in df_kau.groupby(["width", "height"]):
            if len(group) < 20:
                continue
            cluster_balance = group[label_col].mean()
            diff = abs(cluster_balance - overall_balance)
            if diff > max_allowed_diff:
                confounds.append(
                    f"Cluster ({w}x{h}, n={len(group)}): class balance={cluster_balance:.3f} "
                    f"vs overall={overall_balance:.3f} (diff={diff:.3f} > {max_allowed_diff})"
                )

    if confounds:
        err_msg = (
            "KAU-BCMD FORMAT/LABEL CONFOUND DETECTED:\n  " +
            "\n  ".join(confounds) +
            "\nClass label is confounded with image format. Aborting before feature extraction."
        )
        print(f"  [ERROR] {err_msg}")
        raise ValueError(err_msg)

    print("  [Integrity Audit] PASS: No format/label confound detected across resolution clusters.")


def collect_kau_paths(birad_map: Optional[dict] = None, kau_root: Optional[Path] = None) -> pd.DataFrame:
    """Collect KAU-BCMD images, preferring pre-validated eda_outputs/kau_valid_paths.json."""
    valid_json = BASE / "eda_outputs" / "kau_valid_paths.json"
    if valid_json.exists():
        try:
            with open(valid_json, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["label"] = df["label_int"]
            print(f"  [KAU Ingestion] Loaded {len(df)} pre-validated images from {valid_json.name}")
            assert_no_format_confound(df)
            return df[["path", "label"]]
        except Exception as e:
            print(f"  [WARNING] Could not load {valid_json}: {e}. Falling back to disk discovery.")

    k_root = kau_root or ROOT_KAU
    meta_csv = find_kau_metadata(k_root)
    if meta_csv is not None:
        try:
            print(f"  [KAU Ingestion] Ingesting from metadata: {meta_csv}")
            meta = pd.read_csv(meta_csv)
            col_map = {c.strip().lower(): c for c in meta.columns}
            assess_col = next(col_map[c] for c in ["assessment", "birads", "bi-rads", "birad", "class"] if c in col_map)
            path_col = next(col_map[c] for c in ["images path", "image path", "images_path", "image_path", "path"] if c in col_map)
            birad_to_label = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
            records = []
            for _, row in meta.iterrows():
                try:
                    raw_val = row[assess_col]
                    grade = int(raw_val) if not isinstance(raw_val, str) else int(__import__('re').search(r'(\d+)', raw_val).group(1))
                except Exception:
                    continue
                if grade not in birad_to_label:
                    continue
                lbl = birad_to_label[grade]
                rel_p = str(row[path_col]).replace("\\", "/")
                cand = None
                for c in [k_root / rel_p, k_root.parent / rel_p, Path(rel_p)]:
                    if c.exists() and c.is_file():
                        cand = c; break
                if cand is None:
                    matches = list(k_root.glob(f"**/{Path(rel_p).name}"))
                    if matches: cand = matches[0]
                if cand is None or is_kau_mask_path(cand):
                    continue
                try:
                    with Image.open(cand) as img:
                        if img.mode == "1" or (img.width / img.height if img.height else 0) > 1.30:
                            continue
                        records.append({"path": str(cand), "label": lbl, "width": img.width, "height": img.height})
                except Exception:
                    continue
            df = pd.DataFrame(records)
            if not df.empty:
                assert_no_format_confound(df)
                return df[["path", "label"]]
        except Exception as e:
            print(f"  [WARNING] Metadata extraction failed: {e}. Falling back to directory scan.")

    # Fallback directory scan
    benign_dirs = []
    malignant_dirs = []
    for d in k_root.rglob("*"):
        if not d.is_dir() or is_kau_mask_path(d):
            continue
        dn = d.name.lower()
        if any(dn == b for b in ["b1", "b2", "b3", "birad1", "birad2", "birad3", "birads_1", "birads_2", "birads_3"]):
            benign_dirs.append(d)
        elif any(dn == m for m in ["b4", "b5", "birad4", "birad5", "birads_4", "birads_5"]):
            malignant_dirs.append(d)

    if birad_map:
        for p in birad_map.get(0, []):
            if p.exists() and p not in benign_dirs: benign_dirs.append(p)
        for p in birad_map.get(1, []):
            if p.exists() and p not in malignant_dirs: malignant_dirs.append(p)

    records = []
    for label_int, dirs in [(0, benign_dirs), (1, malignant_dirs)]:
        seen = set()
        for directory in dirs:
            for f in sorted(directory.rglob("*")):
                if f.suffix.lower() not in SUPPORTED_EXT or f in seen or is_kau_mask_path(f):
                    continue
                seen.add(f)
                try:
                    with Image.open(f) as img:
                        if img.mode == "1" or (img.width / img.height if img.height else 0) > 1.30:
                            continue
                        records.append({"path": str(f), "label": label_int, "width": img.width, "height": img.height})
                except Exception:
                    pass

    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"No valid images found for KAU-BCMD under {k_root}.")
    assert_no_format_confound(df)
    return df[["path", "label"]]


class MammogramDataset(Dataset):
    """Minimal dataset for feature extraction — no augmentation."""
    def __init__(self, df: pd.DataFrame, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

extract_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTOR: backbone-aware, matches 2a_baseline.py BACKBONE
# ══════════════════════════════════════════════════════════════════════════════

from torchvision.models import (
    ResNet50_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights
)

class FeatureExtractor(nn.Module):
    """
    Strips the classification head from the trained backbone and returns
    the penultimate-layer feature vector for each image.
    Set BACKBONE above to match 2a_baseline.py.

    Output dimensions:
      mobilenetv2    → 1280-dim
      efficientnet_b0 → 1280-dim
      resnet50       → 2048-dim
    """
    def __init__(self, backbone: str, checkpoint_path: Path):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "mobilenetv2":
            base = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=0.3), nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 2)
            )
            self._load(base, checkpoint_path)
            # Feature extractor = everything up to (not including) classifier
            self.extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
            self.feat_dim  = feat_dim

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=0.3), nn.Linear(feat_dim, 128), nn.ReLU(), nn.Linear(128, 2)
            )
            self._load(base, checkpoint_path)
            self.extractor = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
            self.feat_dim  = feat_dim

        elif backbone == "resnet50":
            base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(p=0.4), nn.Linear(feat_dim, 256),
                nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, 2)
            )
            self._load(base, checkpoint_path)
            self.extractor = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
                base.avgpool,
            )
            self.feat_dim = feat_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _load(self, base, checkpoint_path):
        if checkpoint_path.exists():
            base.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print(f"  Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"  [WARNING] Checkpoint not found: {checkpoint_path}")
            print("  Using raw ImageNet weights. Run 2a_baseline.py first.")

    def forward(self, x):
        return self.extractor(x).flatten(1)


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device) -> tuple:
    """
    Run the full DataLoader through the feature extractor.
    Returns (features_np, labels_np) as NumPy arrays.
    """
    model.eval()
    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc="  Extracting", leave=False):
        images = images.to(device)
        feats  = model(images).cpu().numpy()
        all_feats.append(feats)
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PCA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def fit_pca_pipeline(features_train: np.ndarray, n_components: int):
    """
    Fit StandardScaler + PCA on training features.
    CRITICAL: fitted only on training data to prevent leakage.
    Returns the fitted (scaler, pca) tuple.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_train)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"    PCA {n_components} components: "
          f"cumulative explained variance = {cumvar[-1]*100:.2f}%")
    return scaler, pca


def apply_pca_pipeline(scaler, pca, features: np.ndarray) -> np.ndarray:
    scaled = scaler.transform(features)
    return pca.transform(scaled)


def save_pca_pipeline(scaler, pca, n_components: int):
    pipeline = {"scaler": scaler, "pca": pca}
    path = OUT_DIR / f"pca_{n_components}_components.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"    Saved PCA pipeline: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PCA ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_analysis(features_train_raw: np.ndarray, labels_train: np.ndarray):
    """
    Two panels:
      Left  — Scree plot: cumulative explained variance vs number of components
      Right — 2D PCA scatter coloured by class (first 2 PCs)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_train_raw)

    # Fit PCA with max components to get full scree plot
    pca_full = PCA(n_components=min(50, features_train_raw.shape[0],
                                    features_train_raw.shape[1]), random_state=42)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA Analysis — MobileNetV2 Features (2048-dim → compressed)\n"
                 "Mendeley Training Split", fontsize=13, fontweight="bold")

    # ── Scree plot ──
    ax = axes[0]
    ax.plot(range(1, len(cum_var) + 1), cum_var, color="#4C72B0", lw=2, marker="o", markersize=3)
    for n in PCA_N_COMPONENTS:
        ax.axvline(n, color="red", linestyle="--", alpha=0.6, label=f"n={n}: {cum_var[n-1]:.1f}%")
        ax.axhline(cum_var[n-1], color="red", linestyle=":", alpha=0.3)
    ax.axhline(95, color="green", linestyle="-.", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Scree Plot — Cumulative Explained Variance")
    ax.legend(fontsize=8)
    ax.set_xlim([1, min(50, len(cum_var))])
    ax.set_ylim([0, 101])

    # ── 2D PCA scatter ──
    pca_2d  = PCA(n_components=2, random_state=42)
    X_2d    = pca_2d.fit_transform(X_scaled)
    ax = axes[1]
    colors = {0: "#4C72B0", 1: "#DD8452"}
    class_names = {0: "Benign", 1: "Malignant"}
    for label_int, color in colors.items():
        mask = labels_train == label_int
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, alpha=0.6, s=20, label=class_names[label_int], edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("2D PCA Projection — Training Features")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_analysis.png", dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {OUT_DIR / 'pca_analysis.png'}")


def plot_tsne_separability(features_train_raw: np.ndarray, labels_train: np.ndarray,
                           features_train_pca4: np.ndarray):
    """
    t-SNE visualisation comparing separability:
      Left  — t-SNE on full 2048-dim features
      Right — t-SNE on 4-component PCA features
    This validates that the PCA compression doesn't destroy class structure.
    """
    print("  Computing t-SNE (this takes ~1–2 min)...")
    n_sample = min(200, len(labels_train))
    idx      = np.random.choice(len(labels_train), n_sample, replace=False)

    # t-SNE on raw (subsample first for speed)
    scaler   = StandardScaler()
    raw_sub  = scaler.fit_transform(features_train_raw[idx])
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_raw  = tsne_raw.fit_transform(raw_sub)

    # t-SNE on PCA-4
    tsne_pca = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_pca  = tsne_pca.fit_transform(features_train_pca4[idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE: Class Separability Before and After PCA Compression\n"
                 "(n=200 subsample from training split)", fontsize=12, fontweight="bold")
    colors     = {0: "#4C72B0", 1: "#DD8452"}
    class_names = {0: "Benign", 1: "Malignant"}
    labels_sub = labels_train[idx]

    for ax, emb, title in [
        (axes[0], emb_raw, "t-SNE of raw 2048-dim MobileNetV2 features"),
        (axes[1], emb_pca, "t-SNE of 4-component PCA features\n(quantum encoding input)"),
    ]:
        for lbl, color in colors.items():
            mask = labels_sub == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, alpha=0.7, s=25, label=class_names[lbl], edgecolors="none")
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_separability_tsne.png", dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {OUT_DIR / 'feature_separability_tsne.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LINEAR PROBE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_pca_with_linear_probe(features_train_pca: dict, labels_train: np.ndarray,
                                   features_val_pca: dict, labels_val: np.ndarray):
    """
    Quick sanity check: fit a logistic regression on PCA features and
    evaluate on val set.
    Reports both ROC-AUC (primary, threshold-independent) and PR-AUC/average precision
    (secondary, sensitive to minority malignant class) to verify class separability.
    """
    print("\n  ── Linear Probe Validation (PCA features → Logistic Regression) ──")
    results = {}
    for n in PCA_N_COMPONENTS:
        X_tr = features_train_pca[n]
        X_vl = features_val_pca[n]
        # Priority 5: confirm fresh arrays and distinct dimensions per iteration
        tr_hash = hash(X_tr.tobytes()[:500]) if hasattr(X_tr, "tobytes") else "N/A"
        vl_hash = hash(X_vl.tobytes()[:500]) if hasattr(X_vl, "tobytes") else "N/A"
        print(f"    [Diagnostic] PCA n={n}: X_tr shape={X_tr.shape}, id={id(X_tr)}, hash={tr_hash}; "
              f"X_vl shape={X_vl.shape}, id={id(X_vl)}, hash={vl_hash}")
        lr   = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(X_tr, labels_train)
        val_probs = lr.predict_proba(X_vl)[:, 1]
        val_preds = lr.predict(X_vl)
        auc  = roc_auc_score(labels_val, val_probs) if len(set(labels_val)) > 1 else 0.0
        aupr = average_precision_score(labels_val, val_probs) if len(set(labels_val)) > 1 else 0.0
        acc  = accuracy_score(labels_val, val_preds)
        print(f"    PCA n={n}: Val AUC = {auc:.4f}  Val PR-AUC = {aupr:.4f}  Val Acc = {acc:.4f}")
        results[n] = {"val_auc": round(auc, 4), "val_aupr": round(aupr, 4), "val_acc": round(acc, 4)}
    return results


def validate_compression_ablation(X_train_raw: np.ndarray, y_train: np.ndarray,
                                  X_val_raw: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Tier 2 Item 11: Feature compression ablation comparing PCA vs. LDA vs. UMAP.
    Linear probe evaluation on validation set across matched dimension targets.
    """
    print("\n  ── Feature Compression Ablation: PCA vs. LDA vs. UMAP (Linear Probe) ──")
    ablation_results = {}

    # 1. Standard Scaler on raw features
    scaler_raw = StandardScaler().fit(X_train_raw)
    X_tr_sc = scaler_raw.transform(X_train_raw)
    X_vl_sc = scaler_raw.transform(X_val_raw)

    # 2. PCA at matched dimensions
    for n in PCA_N_COMPONENTS:
        pca = PCA(n_components=n, random_state=42).fit(X_tr_sc)
        X_tr_p = pca.transform(X_tr_sc)
        X_vl_p = pca.transform(X_vl_sc)
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0).fit(X_tr_p, y_train)
        probs = lr.predict_proba(X_vl_p)[:, 1]
        auc = roc_auc_score(y_val, probs) if len(set(y_val)) > 1 else 0.0
        aupr = average_precision_score(y_val, probs) if len(set(y_val)) > 1 else 0.0
        ablation_results[f"PCA_n{n}"] = {"method": "PCA", "dim": n, "val_auc": round(auc, 4), "val_aupr": round(aupr, 4)}
        print(f"    PCA  (n={n}): Val AUC = {auc:.4f} | Val PR-AUC = {aupr:.4f}")

    # 3. LDA (yields 1 discriminant direction for binary classification)
    lda = LinearDiscriminantAnalysis().fit(X_tr_sc, y_train)
    X_tr_lda = lda.transform(X_tr_sc)
    X_vl_lda = lda.transform(X_vl_sc)
    lr_lda = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_lda, y_train)
    probs_lda = lr_lda.predict_proba(X_vl_lda)[:, 1]
    auc_lda = roc_auc_score(y_val, probs_lda) if len(set(y_val)) > 1 else 0.0
    aupr_lda = average_precision_score(y_val, probs_lda) if len(set(y_val)) > 1 else 0.0
    ablation_results["LDA_1D"] = {"method": "LDA", "dim": 1, "val_auc": round(auc_lda, 4), "val_aupr": round(aupr_lda, 4)}
    print(f"    LDA  (dim=1): Val AUC = {auc_lda:.4f} | Val PR-AUC = {aupr_lda:.4f} (Supervised Reference)")

    # 4. UMAP (auto-install if missing)
    try:
        import umap
    except ImportError:
        try:
            import subprocess, sys
            print("    [INFO] umap-learn not found. Auto-installing umap-learn...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
            import umap
        except Exception as e:
            print(f"    [WARN] Could not auto-install umap-learn: {e}. Skipping UMAP linear probe comparison.")
            umap = None

    if 'umap' in locals() and umap is not None:
        try:
            for n in PCA_N_COMPONENTS:
                reducer = umap.UMAP(n_components=n, random_state=42, n_neighbors=15, min_dist=0.1)
                X_tr_u = reducer.fit_transform(X_tr_sc)
                X_vl_u = reducer.transform(X_vl_sc)
                lr_u = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_u, y_train)
                probs_u = lr_u.predict_proba(X_vl_u)[:, 1]
                auc_u = roc_auc_score(y_val, probs_u) if len(set(y_val)) > 1 else 0.0
                aupr_u = average_precision_score(y_val, probs_u) if len(set(y_val)) > 1 else 0.0
                ablation_results[f"UMAP_n{n}"] = {"method": "UMAP", "dim": n, "val_auc": round(auc_u, 4), "val_aupr": round(aupr_u, 4)}
                print(f"    UMAP (n={n}): Val AUC = {auc_u:.4f} | Val PR-AUC = {aupr_u:.4f}")
        except Exception as e:
            print(f"    [WARN] UMAP calculation error: {e}")

    return ablation_results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  QUANTUM ENCODING RANGE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_encoding_range(features_pca: np.ndarray, n_components: int):
    """
    Angle encoding maps each feature x_i to θ_i = 2π·x_i.
    For this to work correctly without aliasing, the features should be
    scaled to [0, 1] or [-1, 1] AFTER PCA.
    This function checks the range and prints a warning if rescaling is needed.
    """
    mins  = features_pca.min(axis=0)
    maxes = features_pca.max(axis=0)
    print(f"\n  ── Quantum Encoding Range Check (PCA n={n_components}) ──")
    print(f"    Feature value range: [{mins.min():.3f}, {maxes.max():.3f}]")
    if mins.min() < -1.0 or maxes.max() > 1.0:
        print("    [!] Features exceed [-1, 1] range.")
        print("    → Apply MinMaxScaler to [0, 1] BEFORE angle encoding in the VQC pipeline.")
        print("    → This is already included in 3_vqc.py preprocessing.")
    else:
        print("    ✓ Features within [-1, 1]. Angle encoding safe to apply directly.")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print("  2B — FEATURE EXTRACTION + PCA")
    print("  QML Breast Cancer Classification | Bridge to Quantum Pipeline")
    print("═"*70)

    # ── Cache guard ──────────────────────────────────────────────────────────
    from cache_check import already_done, CACHE
    if already_done("features"):
        return
    # ────────────────────────────────────────────────────────────────────────

    # ── Load split indices ───────────────────────────────────────────────────
    print("\n[1/7] Loading split indices...")
    df_all = collect_paths(MENDELEY_BENIGN, MENDELEY_MALIGNANT)

    if SPLIT_INDEX_FILE.exists():
        with open(SPLIT_INDEX_FILE) as f:
            split_info = json.load(f)
        train_df = df_all.iloc[split_info["train_indices"]].reset_index(drop=True)
        val_df   = df_all.iloc[split_info["val_indices"]].reset_index(drop=True)
        test_df  = df_all.iloc[split_info["test_indices"]].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        labels = df_all["label"].values
        tr_idx, tmp_idx = train_test_split(range(len(df_all)), test_size=0.30,
                                            stratify=labels, random_state=42)
        vl_idx, ts_idx  = train_test_split(tmp_idx, test_size=0.50,
                                            stratify=labels[list(tmp_idx)], random_state=42)
        train_df = df_all.iloc[list(tr_idx)].reset_index(drop=True)
        val_df   = df_all.iloc[list(vl_idx)].reset_index(drop=True)
        test_df  = df_all.iloc[list(ts_idx)].reset_index(drop=True)

    df_kau = collect_kau_paths(KAU_BIRAD_MAP)
    print(f"  Mendeley — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  KAU-BCMD — External validation: {len(df_kau)}")

    # ── Build DataLoaders ────────────────────────────────────────────────────
    print("\n[2/7] Building DataLoaders (no augmentation for feature extraction)...")
    loaders = {
        "train": DataLoader(MammogramDataset(train_df, extract_transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True),
        "val":   DataLoader(MammogramDataset(val_df,   extract_transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True),
        "test":  DataLoader(MammogramDataset(test_df,  extract_transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True),
        "kau":   DataLoader(MammogramDataset(df_kau,   extract_transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True),
    }

    # ── Build feature extractor ──────────────────────────────────────────────
    print("\n[3/7] Loading feature extractor...")
    extractor = FeatureExtractor(BACKBONE, BEST_CHECKPOINT).to(DEVICE)
    print(f"  Backbone: {BACKBONE} | Feature dim: {extractor.feat_dim}")

    # ── Extract raw 2048-dim features ────────────────────────────────────────
    print("\n[4/7] Extracting raw features (2048-dim)...")
    raw_features, raw_labels = {}, {}
    for split, loader in loaders.items():
        print(f"  Processing: {split}...")
        feats, labels = extract_features(extractor, loader, DEVICE)
        raw_features[split] = feats
        raw_labels[split]   = labels
        np.save(OUT_DIR / f"features_{split}_raw.npy", feats)
        np.save(OUT_DIR / f"labels_{split}.npy",       labels)
        print(f"    {split}: {feats.shape}")

    # ── PCA analysis plot ────────────────────────────────────────────────────
    print("\n[5/7] PCA analysis...")
    plot_pca_analysis(raw_features["train"], raw_labels["train"])

    # ── Fit PCA pipelines (train only) and transform all splits ─────────────
    print("\n[6/7] Fitting PCA pipelines and transforming all splits...")
    pca_features = {n: {} for n in PCA_N_COMPONENTS}
    probe_report = {}
    pca_report   = {}

    for n in PCA_N_COMPONENTS:
        print(f"\n  ── PCA n={n} ──")
        scaler, pca = fit_pca_pipeline(raw_features["train"], n)
        save_pca_pipeline(scaler, pca, n)
        pca_report[n] = {
            "n_components": n,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        }
        for split in ["train", "val", "test", "kau"]:
            transformed = apply_pca_pipeline(scaler, pca, raw_features[split])
            pca_features[n][split] = transformed
            np.save(OUT_DIR / f"features_{split}_pca{n}.npy", transformed)

    # ── Quantum encoding range check ─────────────────────────────────────────
    check_encoding_range(pca_features[4]["train"], n_components=4)

    # ── Linear probe validation ──────────────────────────────────────────────
    train_pca_by_n = {n: pca_features[n]["train"] for n in PCA_N_COMPONENTS}
    val_pca_by_n   = {n: pca_features[n]["val"]   for n in PCA_N_COMPONENTS}
    probe_results  = validate_pca_with_linear_probe(
        train_pca_by_n, raw_labels["train"],
        val_pca_by_n,   raw_labels["val"]
    )

    # ── t-SNE separability plot ──────────────────────────────────────────────
    print("\n[7/7] Plotting t-SNE separability...")
    plot_tsne_separability(
        raw_features["train"], raw_labels["train"],
        pca_features[4]["train"]
    )

    # ── Compression ablation (PCA vs LDA vs UMAP) ───────────────────────────
    compression_ablation = validate_compression_ablation(
        raw_features["train"], raw_labels["train"],
        raw_features["val"],   raw_labels["val"]
    )

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "pca_variants":         pca_report,
        "linear_probe":         probe_results,
        "compression_ablation": compression_ablation,
        "feature_shape_raw":    list(raw_features["train"].shape),
        "splits": {
            "train": int(len(raw_labels["train"])),
            "val":   int(len(raw_labels["val"])),
            "test":  int(len(raw_labels["test"])),
            "kau":   int(len(raw_labels["kau"])),
        },
        "note": (
            "PCA fitted on training split only (no data leakage). "
            "MinMaxScaler to [0,1] should be applied inside the VQC pipeline "
            "before angle encoding (not here, to keep PCA features clean). "
            "The recommended starting qubit count for the VQC is 4 (PCA-4). "
            "Extend to 6 or 8 if explained variance is insufficient."
        )
    }
    with open(OUT_DIR / "pca_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n")
    print("═"*70)
    print("  FEATURE EXTRACTION + PCA COMPLETE")
    print("  Outputs saved to:", OUT_DIR)
    print("\n  Files ready for quantum pipeline (3_vqc.py):")
    for n in PCA_N_COMPONENTS:
        cv = pca_report[n]["cumulative_variance"] * 100
        pr = probe_results[n]["val_auc"]
        print(f"    PCA n={n}: {cv:.1f}% var explained | Linear probe AUC={pr:.4f}")
    print("\n  Recommended starting configuration for VQC:")
    print("    → Use PCA n=4 features (lowest qubit cost, fastest simulation)")
    print("    → Apply MinMaxScaler([0,1]) to PCA features before angle encoding")
    print("    → Angle encoding: θ_i = 2π × x_i on 4 qubits")
    print("    → Increase to n=6 if VQC validation AUC < classical baseline")
    print("═"*70)
    CACHE.mark_done("features")


if __name__ == "__main__":
    main()