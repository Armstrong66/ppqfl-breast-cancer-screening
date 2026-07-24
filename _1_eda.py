"""
=============================================================================
1 — DATA AUDIT & EXPLORATORY DATA ANALYSIS

Datasets: (1) Mendeley Mammogram Dataset — Polokwane, South Africa
              DOI: 10.17632/88vzgys5vg.2
          (2) KAU-BCMD — King Abdulaziz University, Saudi Arabia
              https://www.kaggle.com/asmaasaad/king-abdulaziz-university-mammogram-dataset
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import os, warnings, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageStat
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  —  Auto-detect paths to dataset directories
# ═════════════════════════════════════════════════════════════════════════════

def find_mendeley_root() -> Path:
    """Auto-detect the Mendeley dataset root directory.

    Searches relative to this script's location for the dataset.
    The dataset should be located at:
      - scripts/Breast Cancer Dataset/Breast Cancer Original (relative to script)
      - parent of scripts/Breast Cancer Dataset/Breast Cancer Original
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

def find_kau_root() -> Path:
    """Auto-detect the KAU-BCMD dataset root directory.

    Searches relative to this script's location for the dataset.
    """
    script_dir = Path(__file__).resolve().parent
    relative_paths = [
        script_dir / "kau",
        script_dir.parent / "kau",
        Path("./kau"),
        Path("../kau"),
    ]
    for p in relative_paths:
        if p.exists() and (p / "BIRAD1").exists():
            return p.resolve()
    raise FileNotFoundError(
        f"KAU-BCMD dataset not found. Searched relative paths:\n"
        f"  " + "\n  ".join(str(p) for p in relative_paths) + "\n"
        f"Ensure the dataset is located at one of these paths with BIRAD1/ subfolder."
    )

# ── Mendeley (Polokwane, South Africa) ──────────────────────────────────────
ROOT_MENDELEY = find_mendeley_root()
MENDELEY_BENIGN    = ROOT_MENDELEY / "Benign"
MENDELEY_MALIGNANT = ROOT_MENDELEY / "Malignant"

# ── KAU-BCMD ────────────────────────────────────────────────────────────────
# KAU uses BI-RADS grading, not a flat Benign/Malignant structure.
# Binarisation follows clinical convention:
#   BI-RADS 1, 3  → Benign   (label 0)
#   BI-RADS 4, 5  → Malignant (label 1)
ROOT_KAU = find_kau_root()
KAU_BIRAD_MAP = {
    0: [ROOT_KAU / "BIRAD1" / "b1",
        ROOT_KAU / "Birad3" / "b3"],        # Benign
    1: [ROOT_KAU / "Birad4" / "b4",
        ROOT_KAU / "Birad5" / "Birad5"],    # Malignant
}
# Dummy vars kept so downstream calls that pass KAU_BENIGN/KAU_MALIGNANT still work
KAU_BENIGN    = KAU_BIRAD_MAP[0][0]   # used only as a path-existence hint
KAU_MALIGNANT = KAU_BIRAD_MAP[1][0]

PROJECT_ROOT  = Path(__file__).resolve().parent
BASE          = PROJECT_ROOT / "outputs"
OUT_DIR       = BASE / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


def collect_image_paths(benign_dir: Path, malignant_dir: Path,
                        dataset_name: str) -> pd.DataFrame:
    """
    Walk the benign and malignant directories and collect image metadata.
    Exclude binary mask/derivative images from Mendeley at collection time.

    Returns a DataFrame with columns:
        path, label, label_int, filename, ext, dataset
    """
    records = []
    skipped_mask_folder = 0
    skipped_binary_mode = 0
    skipped_mask_paths = []
    skipped_mode_paths = []
    skipped_mode_dims = []
    for label, label_int, directory in [
        ("Benign",    0, benign_dir),
        ("Malignant", 1, malignant_dir),
    ]:
        if not directory.exists():
            print(f"  [WARNING] Directory not found: {directory}")
            print("  → Update the path constants at the top of this script.")
            continue
        files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXT]
        for f in files:
            if is_mask_path(f):
                skipped_mask_folder += 1
                skipped_mask_paths.append(f)
                continue
            try:
                with Image.open(f) as img:
                    if img.mode == "1":
                        skipped_binary_mode += 1
                        skipped_mode_paths.append(f)
                        skipped_mode_dims.append((img.width, img.height))
                        continue
            except Exception:
                # Keep corrupt/unreadable files for audit reporting later.
                pass

            records.append({
                "path":      str(f),
                "label":     label,
                "label_int": label_int,
                "filename":  f.name,
                "ext":       f.suffix.lower(),
                "dataset":   dataset_name,
            })
    if skipped_mask_folder > 0:
        folders = sorted({p.parent.name for p in skipped_mask_paths})
        folder_counts = Counter(p.parent.name for p in skipped_mask_paths)
        print(f"  Skipped {skipped_mask_folder} files from mask/segmentation folders: {folders}")
        print(f"    Folder counts: {dict(folder_counts)}")
    if skipped_binary_mode > 0:
        folders = sorted({p.parent.name for p in skipped_mode_paths})
        dims = np.array(skipped_mode_dims, dtype=int)
        dim_desc = {
            "min_width": int(dims[:, 0].min()),
            "max_width": int(dims[:, 0].max()),
            "min_height": int(dims[:, 1].min()),
            "max_height": int(dims[:, 1].max()),
            "most_common": Counter(map(tuple, dims)).most_common(3),
        }
        print(f"  Skipped {skipped_binary_mode} binary mask files by image mode: {folders}")
        print(f"    Mode='1' dimensions summary: {dim_desc}")
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(
            f"No images found for {dataset_name}.\n"
            f"Benign dir:    {benign_dir}\n"
            f"Malignant dir: {malignant_dir}\n"
            "Please check your ROOT_* path constants."
        )
    return df


def collect_kau_paths(birad_map: dict, dataset_name: str = "KAU-BCMD") -> pd.DataFrame:
    """
    KAU-BCMD uses BI-RADS grading folders, not flat Benign/Malignant.
    birad_map: {label_int: [list of Path dirs]}
    BI-RADS 1,3 → Benign (0); BI-RADS 4,5 → Malignant (1)
    """
    label_names = {0: "Benign", 1: "Malignant"}
    records = []
    skipped_mask_folder = 0
    skipped_binary_mode = 0
    skipped_mask_paths = []
    skipped_mode_paths = []
    for label_int, dirs in birad_map.items():
        for directory in dirs:
            if not directory.exists():
                print(f"  [WARNING] KAU dir not found: {directory}")
                continue
            files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXT]
            for f in files:
                if is_mask_path(f):
                    skipped_mask_folder += 1
                    skipped_mask_paths.append(f)
                    continue
                try:
                    with Image.open(f) as img:
                        if img.mode == "1":
                            skipped_binary_mode += 1
                            skipped_mode_paths.append(f)
                            continue
                except Exception:
                    pass
                records.append({
                    "path":      str(f),
                    "label":     label_names[label_int],
                    "label_int": label_int,
                    "filename":  f.name,
                    "ext":       f.suffix.lower(),
                    "dataset":   dataset_name,
                    "birad_dir": directory.name,
                })
    if skipped_mask_folder > 0:
        folders = sorted({p.parent.name for p in skipped_mask_paths})
        print(f"  Skipped {skipped_mask_folder} KAU files from mask/segmentation folders: {folders}")
    if skipped_binary_mode > 0:
        folders = sorted({p.parent.name for p in skipped_mode_paths})
        print(f"  Skipped {skipped_binary_mode} KAU binary mask files by image mode: {folders}")
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(
            f"No images found for {dataset_name}.\n"
            f"Check KAU_BIRAD_MAP paths at the top of this script."
        )
    return df


def audit_image(path: str) -> dict:
    """
    Open one image and extract:
        width, height, mode (RGB/L/RGBA), file_size_kb,
        mean_pixel, std_pixel, min_pixel, max_pixel, is_corrupt
    """
    result = {
        "width": None, "height": None, "mode": None,
        "file_size_kb": os.path.getsize(path) / 1024,
        "mean_pixel": None, "std_pixel": None,
        "min_pixel": None,  "max_pixel": None,
        "is_corrupt": False,
    }
    try:
        with Image.open(path) as img:
            result["width"]  = img.width
            result["height"] = img.height
            result["mode"]   = img.mode
            arr = np.array(img.convert("L"), dtype=np.float32)
            result["mean_pixel"] = float(arr.mean())
            result["std_pixel"]  = float(arr.std())
            result["min_pixel"]  = float(arr.min())
            result["max_pixel"]  = float(arr.max())
    except Exception as e:
        result["is_corrupt"] = True
        print(f"  [CORRUPT] {path}: {e}")
    return result


def run_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Audit all images in df; append pixel and dimension stats."""
    stats = []
    for path in tqdm(df["path"], desc=f"  Auditing {df['dataset'].iloc[0]}"):
        stats.append(audit_image(path))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(stats)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_balance(df_m: pd.DataFrame, df_k: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Balance — Primary and External Validation Datasets", fontsize=14, fontweight="bold")

    for ax, df, title in [
        (axes[0], df_m, "Mendeley (Polokwane, South Africa)\nPrimary Training Cohort"),
        (axes[1], df_k, "KAU-BCMD (Saudi Arabia)\nExternal Validation Cohort"),
    ]:
        counts = df["label"].value_counts()
        colors = ["#4C72B0", "#DD8452"]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Image Count")
        ax.set_xlabel("Class")
        for bar, val in zip(bars, counts.values):
            pct = val / counts.sum() * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        total = counts.sum()
        ratio = counts.max() / counts.min()
        ax.text(0.97, 0.97, f"Total: {total}\nImbalance ratio: {ratio:.2f}x",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_class_balance.png", dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {OUT_DIR / '01_class_balance.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RESOLUTION & ASPECT RATIO
# ══════════════════════════════════════════════════════════════════════════════

def plot_resolution(df_m: pd.DataFrame, df_k: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Image Resolution Distribution", fontsize=14, fontweight="bold")

    for i, (df, name) in enumerate([(df_m, "Mendeley"), (df_k, "KAU-BCMD")]):
        df = df.dropna(subset=["width", "height"])

        # Width distribution
        axes[i][0].hist(df["width"], bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[i][0].axvline(df["width"].median(), color="red", linestyle="--", label=f"Median: {df['width'].median():.0f}")
        axes[i][0].set_title(f"{name} — Width Distribution")
        axes[i][0].set_xlabel("Pixel Width")
        axes[i][0].set_ylabel("Count")
        axes[i][0].legend()

        # Height distribution
        axes[i][1].hist(df["height"], bins=30, color="#DD8452", edgecolor="white", alpha=0.85)
        axes[i][1].axvline(df["height"].median(), color="navy", linestyle="--", label=f"Median: {df['height'].median():.0f}")
        axes[i][1].set_title(f"{name} — Height Distribution")
        axes[i][1].set_xlabel("Pixel Height")
        axes[i][1].set_ylabel("Count")
        axes[i][1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_resolution_distribution.png", dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {OUT_DIR / '02_resolution_distribution.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PIXEL INTENSITY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def plot_pixel_stats(df_m: pd.DataFrame, df_k: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pixel Intensity Statistics by Class", fontsize=14, fontweight="bold")

    for i, (df, name) in enumerate([(df_m, "Mendeley"), (df_k, "KAU-BCMD")]):
        df_clean = df.dropna(subset=["mean_pixel"])

        # Mean pixel by class (boxplot)
        data_by_class = [
            df_clean[df_clean["label"] == "Benign"]["mean_pixel"].values,
            df_clean[df_clean["label"] == "Malignant"]["mean_pixel"].values,
        ]
        bp = axes[i][0].boxplot(data_by_class, labels=["Benign", "Malignant"],
                                 patch_artist=True, notch=True)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        axes[i][0].set_title(f"{name} — Mean Pixel Intensity by Class")
        axes[i][0].set_ylabel("Mean Pixel Value (Grayscale 0–255)")

        # Std pixel by class (violin)
        axes[i][1].violinplot(
            [df_clean[df_clean["label"] == "Benign"]["std_pixel"].values,
             df_clean[df_clean["label"] == "Malignant"]["std_pixel"].values],
            positions=[1, 2], showmedians=True
        )
        axes[i][1].set_xticks([1, 2])
        axes[i][1].set_xticklabels(["Benign", "Malignant"])
        axes[i][1].set_title(f"{name} — Pixel Std Dev by Class")
        axes[i][1].set_ylabel("Std Dev of Pixel Values")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_pixel_statistics.png", dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {OUT_DIR / '03_pixel_statistics.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAMPLE IMAGE GRID
# ══════════════════════════════════════════════════════════════════════════════

def plot_sample_grid(df: pd.DataFrame, dataset_name: str, n_per_class: int = 4):
    """Display n_per_class random samples from each class side-by-side."""
    fig, axes = plt.subplots(2, n_per_class, figsize=(4 * n_per_class, 9))
    fig.suptitle(f"Sample Images — {dataset_name}", fontsize=13, fontweight="bold")

    for row, label in enumerate(["Benign", "Malignant"]):
        subset = df[df["label"] == label].sample(
            min(n_per_class, len(df[df["label"] == label])), random_state=42
        )
        for col, (_, row_data) in enumerate(subset.iterrows()):
            ax = axes[row][col]
            try:
                img = Image.open(row_data["path"]).convert("L")
                ax.imshow(img, cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "Load\nError", ha="center", va="center")
            ax.set_title(f"{label}\n{row_data['width']}×{row_data['height']}",
                         fontsize=8, color="#CC3300" if label == "Malignant" else "#004499")
            ax.axis("off")
        # Label the row
        axes[row][0].set_ylabel(label, fontsize=12, fontweight="bold", rotation=0,
                                labelpad=50, va="center")

    plt.tight_layout()
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    savepath = OUT_DIR / f"04_sample_grid_{safe_name}.png"
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {savepath}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PIXEL HISTOGRAM (mean intensity distribution across dataset)
# ══════════════════════════════════════════════════════════════════════════════

def plot_intensity_histogram(df_m: pd.DataFrame, df_k: pd.DataFrame):
    """
    For a random subsample, compute the actual pixel histogram (not just mean)
    to understand dataset-level intensity distribution. This is important for
    deciding on normalisation strategy before quantum encoding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset-Level Pixel Intensity Distribution (Grayscale)\n"
                 "Critical for normalisation before PCA → quantum encoding",
                 fontsize=12, fontweight="bold")

    for ax, df, name in [(axes[0], df_m, "Mendeley (South Africa)"),
                          (axes[1], df_k, "KAU-BCMD (Saudi Arabia)")]:
        sample = df.sample(min(50, len(df)), random_state=42)
        all_pixels = []
        for _, row in sample.iterrows():
            try:
                arr = np.array(Image.open(row["path"]).convert("L").resize((128, 128)),
                               dtype=np.float32).flatten()
                all_pixels.append(arr)
            except Exception:
                continue
        if all_pixels:
            combined = np.concatenate(all_pixels)
            ax.hist(combined, bins=100, color="#4C72B0", alpha=0.75, edgecolor="none",
                    density=True, label="All pixels")
            ax.axvline(combined.mean(), color="red", linestyle="--",
                       label=f"Mean: {combined.mean():.1f}")
            ax.axvline(np.percentile(combined, 5),  color="orange", linestyle=":",
                       label=f"P5: {np.percentile(combined, 5):.1f}")
            ax.axvline(np.percentile(combined, 95), color="green", linestyle=":",
                       label=f"P95: {np.percentile(combined, 95):.1f}")
        ax.set_title(f"{name}\n(n=50 random images, 128×128 pixels each)")
        ax.set_xlabel("Pixel Intensity (0–255)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_pixel_intensity_histogram.png", dpi=150, bbox_inches="tight")
    # plt.show()  # disabled: headless server
    print(f"  Saved: {OUT_DIR / '05_pixel_intensity_histogram.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FILE FORMAT & CORRUPTION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_corruption_report(df_m_audit: pd.DataFrame, df_k_audit: pd.DataFrame):
    print("\n" + "═"*60)
    print("  CORRUPTION & FORMAT AUDIT REPORT")
    print("═"*60)
    for df, name in [(df_m_audit, "Mendeley"), (df_k_audit, "KAU-BCMD")]:
        corrupt = df["is_corrupt"].sum()
        ext_counts = df["ext"].value_counts().to_dict()
        mode_counts = df["mode"].value_counts().to_dict() if "mode" in df.columns else {}
        print(f"\n  {name}:")
        print(f"    Total images  : {len(df)}")
        print(f"    Corrupt/unread: {corrupt}")
        print(f"    File formats  : {ext_counts}")
        print(f"    Colour modes  : {mode_counts}")
        if "width" in df.columns:
            unique_res = df.dropna(subset=["width","height"]).groupby(["width","height"]).size()
            print(f"    Unique resolutions: {len(unique_res)}")
            print(f"    Most common: {unique_res.idxmax()} ({unique_res.max()} images)")
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SUMMARY STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(df_m_audit: pd.DataFrame, df_k_audit: pd.DataFrame):
    print("\n" + "═"*70)
    print("  PIXEL STATISTICS SUMMARY (mean over all images, by class)")
    print("═"*70)
    for df, name in [(df_m_audit, "Mendeley"), (df_k_audit, "KAU-BCMD")]:
        print(f"\n  {name}:")
        summary = df.groupby("label")[["mean_pixel","std_pixel","file_size_kb"]].describe().round(2)
        print(summary.to_string())

    print("\n  Note: Mean pixel values differ between classes can indicate contrast")
    print("  differences between benign and malignant tissue — a useful sanity check")
    print("  that labels are radiologically plausible before model training.")
    print("═"*70)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SAVE AUDIT DATAFRAMES
# ══════════════════════════════════════════════════════════════════════════════

def save_audit_csvs(df_m_audit: pd.DataFrame, df_k_audit: pd.DataFrame):
    mendeley_path = OUT_DIR / "mendeley_audit.csv"
    kau_path      = OUT_DIR / "kau_audit.csv"
    df_m_audit.to_csv(mendeley_path, index=False)
    df_k_audit.to_csv(kau_path,      index=False)
    print(f"\n  Audit CSVs saved:")
    print(f"    {mendeley_path}")
    print(f"    {kau_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRAIN / VAL / TEST SPLIT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_split_plan(df_m_audit: pd.DataFrame):
    """
    Plan the stratified split for the Mendeley dataset before any model training.
    Recommended: 70 / 15 / 15 stratified by class.
    The split itself is done in 2a_baseline.py; this just previews counts.
    """
    from sklearn.model_selection import train_test_split

    df_clean = df_m_audit[~df_m_audit["is_corrupt"]].reset_index(drop=True)
    labels   = df_clean["label_int"].values

    train_idx, temp_idx = train_test_split(
        range(len(df_clean)), test_size=0.30, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=labels[list(temp_idx)],
        random_state=42,
    )

    print("\n" + "═"*60)
    print("  PROPOSED TRAIN / VAL / TEST SPLIT (Mendeley)")
    print("  Stratified split — 70 / 15 / 15")
    print("═"*60)
    for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        subset = df_clean.iloc[list(idx)]
        b = (subset["label"] == "Benign").sum()
        m = (subset["label"] == "Malignant").sum()
        print(f"  {split_name:5s}: {len(subset):4d} images  |  Benign: {b}  Malignant: {m}")
    print("  KAU-BCMD: used entirely as external validation (no split)")
    print("═"*60)

    # Save split index for reproducibility
    split_info = {
        "train_indices": list(train_idx),
        "val_indices":   list(val_idx),
        "test_indices":  list(test_idx),
        "random_state":  42,
        "split_ratio":   "70/15/15",
        "strategy":      "stratified by class label",
    }
    with open(OUT_DIR / "mendeley_split_indices.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Split indices saved to: {OUT_DIR / 'mendeley_split_indices.json'}")
    print("  Import this in 2a_baseline.py for fully reproducible training.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═"*70)
    print("  1 — DATA AUDIT & EDA")
    print("  QFL Breast Cancer Classification | African & MENA Populations")
    print("═"*70)

    # ── Cache guard — skip if all sentinel files already exist ───────────────
    from cache_check import already_done, CACHE
    if already_done("eda"):
        return
    # ────────────────────────────────────────────────────────────────────────

    # ── Step 1: Collect paths ────────────────────────────────────────────────
    print("\n[1/7] Collecting image paths...")
    df_m_paths = collect_image_paths(MENDELEY_BENIGN, MENDELEY_MALIGNANT, "Mendeley")
    df_k_paths = collect_kau_paths(KAU_BIRAD_MAP)
    print(f"  Mendeley: {len(df_m_paths)} images  "
          f"({(df_m_paths['label']=='Benign').sum()} B / "
          f"{(df_m_paths['label']=='Malignant').sum()} M)")
    print(f"  KAU-BCMD: {len(df_k_paths)} images  "
          f"({(df_k_paths['label']=='Benign').sum()} B / "
          f"{(df_k_paths['label']=='Malignant').sum()} M)")

    # ── Step 2: Audit images (read dimensions + pixel stats) ─────────────────
    print("\n[2/7] Auditing images (dimensions, pixel stats, corruption check)...")
    df_m_audit = run_audit(df_m_paths)
    df_k_audit = run_audit(df_k_paths)

    # ── Step 3: Class balance ────────────────────────────────────────────────
    print("\n[3/7] Plotting class balance...")
    plot_class_balance(df_m_audit, df_k_audit)

    # ── Step 4: Resolution distributions ────────────────────────────────────
    print("\n[4/7] Plotting resolution distributions...")
    plot_resolution(df_m_audit, df_k_audit)

    # ── Step 5: Pixel statistics ─────────────────────────────────────────────
    print("\n[5/7] Plotting pixel intensity statistics...")
    plot_pixel_stats(df_m_audit, df_k_audit)
    plot_intensity_histogram(df_m_audit, df_k_audit)

    # ── Step 6: Sample image grids ───────────────────────────────────────────
    print("\n[6/7] Plotting sample image grids...")
    plot_sample_grid(df_m_audit, "Mendeley — Polokwane, South Africa", n_per_class=4)
    plot_sample_grid(df_k_audit, "KAU-BCMD — Saudi Arabia",           n_per_class=4)

    # ── Step 7: Reports, split plan, save CSVs ───────────────────────────────
    print("\n[7/7] Generating reports...")
    print_corruption_report(df_m_audit, df_k_audit)
    print_summary_table(df_m_audit, df_k_audit)
    print_split_plan(df_m_audit)
    save_audit_csvs(df_m_audit, df_k_audit)

    print("\n✓ EDA COMPLETE. All outputs saved to:", OUT_DIR)
    print("  Next step → 2a_baseline.py")
    CACHE.mark_done("eda")


if __name__ == "__main__":
    main()