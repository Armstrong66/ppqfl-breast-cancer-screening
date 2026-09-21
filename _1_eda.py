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
from typing import Optional, List, Dict, Tuple

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

def _looks_like_kau_metadata(path: Path) -> bool:
    """Validate that candidate CSV contains expected KAU-BCMD columns."""
    try:
        df_head = pd.read_csv(path, nrows=2)
        cols = {str(c).strip().lower() for c in df_head.columns}
        has_assessment = any(c in {"assessment", "birads", "bi-rads", "birad", "class", "grade"} for c in cols)
        has_path = any("path" in c or "file" in c for c in cols)
        return has_assessment and has_path
    except Exception:
        return False


def export_kau_xlsx_to_csv(xlsx_path: Path, csv_path: Path) -> bool:
    """Extract sheet containing image paths & assessments from correctSheetlast.xlsx using standard library."""
    import zipfile, xml.etree.ElementTree as ET, csv
    if not xlsx_path.exists():
        return False
    try:
        with zipfile.ZipFile(xlsx_path, 'r') as z:
            shared_strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
                for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    texts = [t.text for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text]
                    shared_strings.append(''.join(texts))

            wb_tree = ET.fromstring(z.read('xl/workbook.xml'))
            sheets = wb_tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet')
            target_sheet_file = None
            for s in sheets:
                s_name = s.attrib.get('name', '').lower()
                if 'correctsheet' in s_name:
                    r_id = s.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                    rels_tree = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
                    for rel in rels_tree.findall('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
                        if rel.attrib.get('Id') == r_id:
                            target_sheet_file = 'xl/' + rel.attrib.get('Target').lstrip('/')
                            break
                    if target_sheet_file:
                        break

            if not target_sheet_file:
                target_sheet_file = 'xl/worksheets/sheet2.xml' if 'xl/worksheets/sheet2.xml' in z.namelist() else 'xl/worksheets/sheet1.xml'

            st_tree = ET.fromstring(z.read(target_sheet_file))
            rows_to_export = []
            for r in st_tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
                row_vals = []
                for c in r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                    t = c.attrib.get('t')
                    v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    val = v.text if v is not None else ''
                    if t == 's' and val.isdigit():
                        idx = int(val)
                        val = shared_strings[idx] if idx < len(shared_strings) else val
                    row_vals.append(val)
                if any(row_vals):
                    rows_to_export.append(row_vals[:8])

            if rows_to_export:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows_to_export)
                print(f"  [KAU Metadata] Auto-extracted {len(rows_to_export)} records from {xlsx_path.name} -> {csv_path.name}")
                return True
    except Exception as e:
        print(f"  [KAU Metadata] Failed to auto-extract from {xlsx_path}: {e}")
    return False


def find_kau_metadata(kau_root: Path) -> Optional[Path]:
    """
    Search for the official KAU-BCMD Metadata.csv, or auto-extract it
    from correctSheetlast.xlsx if present.
    """
    direct = [
        kau_root / "Metadata.csv",
        kau_root / "metadata.csv",
        kau_root.parent / "Metadata.csv",
        kau_root.parent / "metadata.csv",
        Path("/data/derrick/kau/Metadata.csv"),
        Path("/data/derrick/kau/metadata.csv"),
        Path("./kau/Metadata.csv"),
        Path("../kau/Metadata.csv"),
    ]
    for cp in direct:
        if cp.exists() and cp.is_file() and _looks_like_kau_metadata(cp):
            print(f"  [KAU Metadata] Found and verified: {cp.resolve()}")
            return cp.resolve()

    # Search strictly within kau_root
    if kau_root.exists() and kau_root.is_dir():
        for name in ["Metadata.csv", "metadata.csv"]:
            for hit in kau_root.rglob(name):
                if _looks_like_kau_metadata(hit):
                    print(f"  [KAU Metadata] Found and verified via rglob: {hit.resolve()}")
                    return hit.resolve()
        for hit in kau_root.rglob("*[Mm]etadata*.csv"):
            if _looks_like_kau_metadata(hit):
                print(f"  [KAU Metadata] Found and verified via wildcard: {hit.resolve()}")
                return hit.resolve()

    # Auto-extract from correctSheetlast.xlsx if available
    xlsx_candidates = [
        kau_root / "correctSheetlast.xlsx",
        kau_root.parent / "correctSheetlast.xlsx",
        Path("/data/derrick/kau/correctSheetlast.xlsx"),
    ]
    for x in xlsx_candidates:
        if x.exists() and x.is_file():
            target_csv = x.parent / "Metadata.csv"
            if export_kau_xlsx_to_csv(x, target_csv) and _looks_like_kau_metadata(target_csv):
                return target_csv.resolve()

    print(
        f"  [KAU Metadata] WARNING: Valid KAU-BCMD Metadata.csv not found.\n"
        f"  Searched within: {kau_root}\n"
        f"  Manual action required: locate correctSheetlast.xlsx or Metadata.csv in the KAU-BCMD release\n"
        f"  and place it at: {kau_root / 'Metadata.csv'}"
    )
    return None

# ── Mendeley (Polokwane, South Africa) ──────────────────────────────────────
ROOT_MENDELEY = find_mendeley_root()
MENDELEY_BENIGN    = ROOT_MENDELEY / "Benign"
MENDELEY_MALIGNANT = ROOT_MENDELEY / "Malignant"

# ── KAU-BCMD ────────────────────────────────────────────────────────────────
# KAU uses BI-RADS grading, not a flat Benign/Malignant structure.
# Binarisation follows clinical convention:
#   BI-RADS 1, 2, 3 → Benign   (label 0)
#   BI-RADS 4, 5    → Malignant (label 1)
ROOT_KAU = find_kau_root()
KAU_BIRAD_MAP = {
    0: [ROOT_KAU / "BIRAD1" / "b1",
        ROOT_KAU / "Birad3" / "b3"],        # Benign (legacy fallback folders)
    1: [ROOT_KAU / "Birad4" / "b4",
        ROOT_KAU / "Birad5" / "Birad5"],    # Malignant (legacy fallback folders)
}
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
        files = sorted([f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXT], key=lambda p: str(p))
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


def is_kau_mask_path(path: Path) -> bool:
    """
    Folder/filename-based mask exclusion, extending mask hints to also cover
    KAU-specific naming ('Tumor Masks', 'masks', 'segmentation', 'reports', etc.).
    Needed because JPEG masks do not use mode == '1' and evade PIL mode filters.
    """
    hints = {"tumor masks", "tumor mask", "mask", "masks", "segmentation",
             "segmentations", "report", "reports", "ground_truth", "groundtruth", "gt"}
    parts = [part.lower() for part in path.parts[:-1]]
    stem = path.stem.lower()
    if any(any(h in part for h in hints) for part in parts):
        return True
    return any(h in stem for h in hints)


def collect_kau_paths_from_metadata(metadata_csv: Path, kau_root: Path,
                                    dataset_name: str = "KAU-BCMD") -> pd.DataFrame:
    """
    Ingest KAU-BCMD strictly from the official Metadata.csv.
    Every row's image path is verified on disk via stem and path index.
    Non-primary content (masks, reports, segmentations) is filtered out systematically.
    BI-RADS mapping: 1, 2, 3 -> Benign (0); 4, 5 -> Malignant (1).
    """
    print(f"  [KAU Ingestion] Using official metadata: {metadata_csv}")
    meta = pd.read_csv(metadata_csv)
    col_map = {c.strip().lower(): c for c in meta.columns}

    assess_col = None
    for cand in ["assessment", "assesment", "birads", "bi-rads", "birad", "class", "grade"]:
        if cand in col_map:
            assess_col = col_map[cand]
            break
    if assess_col is None:
        raise ValueError(f"Could not find Assessment / BI-RADS column in {metadata_csv}. Columns: {list(meta.columns)}")

    path_col = None
    for cand in ["images path", "image path", "images_path", "image_path", "path", "file_path", "filename", "file"]:
        if cand in col_map:
            path_col = col_map[cand]
            break
    if path_col is None:
        raise ValueError(f"Could not find Images Path column in {metadata_csv}. Columns: {list(meta.columns)}")

    # Pre-index all image files under kau_root by stem for fast, robust cross-format lookup
    file_index: Dict[str, Path] = {}
    for p in kau_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            file_index[p.stem.strip().lower()] = p

    label_names = {0: "Benign", 1: "Malignant"}
    birad_to_label = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

    records = []
    skipped_masks = 0
    skipped_missing = 0

    for _, row in meta.iterrows():
        raw_val = row[assess_col]
        try:
            if isinstance(raw_val, str):
                import re
                m = re.search(r"(\d+)", raw_val)
                grade = int(m.group(1)) if m else None
            else:
                grade = int(raw_val)
        except Exception:
            continue

        if grade not in birad_to_label:
            continue
        label_int = birad_to_label[grade]

        raw_p = str(row[path_col]).strip()
        stem = Path(raw_p).stem.strip().lower()

        candidate_file = file_index.get(stem)
        if candidate_file is None:
            # Fallback direct path check
            rel_p = raw_p.replace("\\", "/")
            for direct in [kau_root / rel_p, kau_root.parent / rel_p, Path(rel_p)]:
                if direct.exists() and direct.is_file():
                    candidate_file = direct
                    break

        if candidate_file is None:
            skipped_missing += 1
            continue

        if is_kau_mask_path(candidate_file):
            skipped_masks += 1
            continue

        try:
            with Image.open(candidate_file) as img:
                if img.mode == "1":
                    continue
                w, h = img.width, img.height
        except Exception:
            continue

        records.append({
            "path":      str(candidate_file),
            "label":     label_names[label_int],
            "label_int": label_int,
            "filename":  candidate_file.name,
            "ext":       candidate_file.suffix.lower(),
            "dataset":   dataset_name,
            "birad_dir": f"BIRADS_{grade}",
            "width":     w,
            "height":    h,
        })

    print(f"  [Metadata Ingestion] Ingested {len(records)} valid images "
          f"(skipped {skipped_masks} masks, {skipped_missing} missing on disk).")
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"Metadata ingestion from {metadata_csv} yielded 0 images.")
    return df


def collect_kau_paths(kau_root: Path,
                      birad_map: Optional[dict] = None,
                      dataset_name: str = "KAU-BCMD") -> pd.DataFrame:
    """
    Collect KAU-BCMD mammograms with dual ingestion strategy:

    1. PRIMARY: If Metadata.csv is found, use collect_kau_paths_from_metadata().
       No aspect-ratio filter — metadata is ground truth for what is a mammogram.

    2. FALLBACK (directory scan): Used ONLY when Metadata.csv is unavailable.
       - Computes per-BI-RADS-folder retention rates.
       - Prints a retention table showing n_before / n_after / retention% per folder.
       - RAISES ValueError if any BI-RADS folder is 100% eliminated (0% retention),
         because that pattern is structurally indistinguishable from a format/label
         confound. The researcher must locate Metadata.csv to proceed safely.
    """
    metadata_csv = find_kau_metadata(kau_root)
    if metadata_csv is not None:
        try:
            return collect_kau_paths_from_metadata(metadata_csv, kau_root, dataset_name)
        except Exception as e:
            print(f"  [WARNING] Metadata-driven ingestion failed ({e}). Falling back to directory crawl.")

    # ── FALLBACK: directory scan ───────────────────────────────────────────────
    print(
        "\n  *** FALLBACK MODE: Metadata.csv not found. Using directory auto-discovery ***\n"
        "  *** Aspect-ratio heuristic (W/H > 1.30) will be applied.                 ***\n"
        "  *** If any BI-RADS folder is 100% eliminated, the pipeline will STOP.    ***\n"
        "  *** Locate Metadata.csv from the KAU-BCMD release to proceed safely.     ***\n"
    )
    label_names = {0: "Benign", 1: "Malignant"}
    benign_dirs = []
    malignant_dirs = []

    for d in kau_root.rglob("*"):
        if not d.is_dir():
            continue
        d_name_lower = d.name.lower()
        if is_kau_mask_path(d):
            continue
        if any(d_name_lower == b for b in [
            "b1", "b2", "b3",
            "birad1", "birad2", "birad3",
            "birads_1", "birads_2", "birads_3",
            "birad_1", "birad_2", "birad_3",
        ]):
            benign_dirs.append(d)
        elif any(d_name_lower == m for m in [
            "b4", "b5",
            "birad4", "birad5",
            "birads_4", "birads_5",
            "birad_4", "birad_5",
        ]):
            malignant_dirs.append(d)

    if birad_map:
        for p in birad_map.get(0, []):
            if p.exists() and p not in benign_dirs:
                benign_dirs.append(p)
        for p in birad_map.get(1, []):
            if p.exists() and p not in malignant_dirs:
                malignant_dirs.append(p)

    all_dirs = [(0, benign_dirs), (1, malignant_dirs)]
    records = []
    folder_stats: Dict[str, Dict] = {}

    skipped_mask_count = 0
    skipped_binary_mode = 0
    skipped_aspect_count = 0

    for label_int, dirs in all_dirs:
        seen_files: set = set()
        for directory in dirs:
            dir_name = directory.name
            files = sorted(
                [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED_EXT],
                key=lambda p: str(p),
            )
            n_before = 0
            n_after = 0
            for f in files:
                if f in seen_files:
                    continue
                seen_files.add(f)
                if is_kau_mask_path(f):
                    skipped_mask_count += 1
                    continue
                n_before += 1
                try:
                    with Image.open(f) as img:
                        if img.mode == "1":
                            skipped_binary_mode += 1
                            continue
                        w, h = img.width, img.height
                        aspect = w / h if h > 0 else 0
                        if aspect > 1.30:
                            skipped_aspect_count += 1
                            continue
                except Exception:
                    pass

                n_after += 1
                records.append({
                    "path":      str(f),
                    "label":     label_names[label_int],
                    "label_int": label_int,
                    "filename":  f.name,
                    "ext":       f.suffix.lower(),
                    "dataset":   dataset_name,
                    "birad_dir": dir_name,
                })

            if dir_name not in folder_stats:
                folder_stats[dir_name] = {"n_before": 0, "n_after": 0, "label_int": label_int}
            folder_stats[dir_name]["n_before"] += n_before
            folder_stats[dir_name]["n_after"] += n_after

    # Per-folder retention table
    print("\n  [Fallback Filter] Per-folder retention after W/H > 1.30 exclusion:")
    print(f"  {'Folder':<20} {'Class':<12} {'Before':>8} {'After':>8} {'Kept%':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    eliminated_folders = []
    for folder, stats in sorted(folder_stats.items()):
        cls = label_names.get(stats["label_int"], "?")
        nb, na = stats["n_before"], stats["n_after"]
        pct = (na / nb * 100) if nb > 0 else 0.0
        marker = "  *** 0% RETAINED ***" if nb > 0 and na == 0 else ""
        print(f"  {folder:<20} {cls:<12} {nb:>8} {na:>8} {pct:>7.1f}%{marker}")
        if nb > 0 and na == 0:
            eliminated_folders.append(folder)

    if skipped_mask_count > 0:
        print(f"  Skipped {skipped_mask_count} KAU mask/report files")
    if skipped_binary_mode > 0:
        print(f"  Skipped {skipped_binary_mode} KAU binary mask mode files")
    if skipped_aspect_count > 0:
        print(f"  Skipped {skipped_aspect_count} KAU non-standard composite/strip images (aspect W/H > 1.30)")

    if eliminated_folders:
        raise ValueError(
            f"\n  [CRITICAL] Fallback aspect-ratio heuristic fully eliminated the following\n"
            f"  BI-RADS folder(s): {eliminated_folders}\n\n"
            f"  This is structurally identical to a format/label confound — the pipeline\n"
            f"  cannot safely continue without verified metadata.\n\n"
            f"  REQUIRED ACTION:\n"
            f"    1. Locate Metadata.csv from the official KAU-BCMD release.\n"
            f"       (Mendeley Data: https://data.mendeley.com/datasets/rnkb4nk9gs)\n"
            f"    2. Place it at: {kau_root / 'Metadata.csv'}\n"
            f"    3. Re-run this script — metadata-driven ingestion will be used instead.\n\n"
            f"  If the eliminated folders genuinely contain non-mammographic images,\n"
            f"  document that finding and use the metadata to confirm it before proceeding."
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(
            f"No valid images found for {dataset_name}.\n"
            f"Searched under {kau_root}."
        )
    return df


def assert_no_format_confound(df_kau: pd.DataFrame, max_allowed_diff: float = 0.25):
    """
    Two-level integrity check after KAU ingestion:

    Level 1 — Per-folder retention (most sensitive):
        For each BI-RADS source folder, report the malignant fraction.
        If benign folders are 100% benign AND malignant folders are 100% malignant
        post-filter, that trivially passes the cluster check but is still the exact
        confound pattern we are looking for.

    Level 2 — Per-resolution-cluster (original check):
        For every (width × height) cluster with ≥20 images, flag if its class
        balance deviates > max_allowed_diff from the overall cohort balance.
    """
    if df_kau.empty:
        raise ValueError("Cannot run assert_no_format_confound on empty dataframe.")

    label_col = "label_int" if "label_int" in df_kau.columns else "label"
    overall_balance = df_kau[label_col].mean()
    n_benign = (df_kau[label_col] == 0).sum()
    n_malignant = (df_kau[label_col] == 1).sum()
    print(f"\n  [Integrity Audit] KAU-BCMD Cohort: {len(df_kau)} total images "
          f"({n_benign} Benign, {n_malignant} Malignant | malignant_rate={overall_balance:.3f})")

    # ── Level 1: per-folder balance ────────────────────────────────────────────
    if "birad_dir" in df_kau.columns:
        print(f"\n  [Integrity Audit] Per-BI-RADS-folder class balance:")
        print(f"  {'Folder':<20} {'n':>6} {'Benign':>8} {'Malignant':>10} {'Mal%':>7}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10} {'-'*7}")
        folder_confounds = []
        for folder, grp in df_kau.groupby("birad_dir"):
            nb = (grp[label_col] == 0).sum()
            nm = (grp[label_col] == 1).sum()
            mal_pct = nm / len(grp) if len(grp) > 0 else 0.0
            print(f"  {str(folder):<20} {len(grp):>6} {nb:>8} {nm:>10} {mal_pct:>6.1%}")
            if "4" in str(folder).lower() or "5" in str(folder).lower():
                if nm == 0 and len(grp) > 0:
                    folder_confounds.append(
                        f"  Folder '{folder}' (n={len(grp)}) is expected malignant "
                        f"but contains 0 malignant images (100% benign after filtering)."
                    )
        if folder_confounds:
            print(
                "\n  [WARN] Per-folder confound signals detected:\n" +
                "\n".join(folder_confounds) +
                "\n  This may indicate the heuristic filter is misclassifying real malignant images."
                "\n  Strongly recommend locating Metadata.csv and using metadata-driven ingestion."
            )

    # ── Level 2: per-resolution-cluster ───────────────────────────────────────
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
            "\nClass label is confounded with image format. Aborting before external validation."
        )
        print(f"  [ERROR] {err_msg}")
        raise ValueError(err_msg)

    print("  [Integrity Audit] PASS: No format/label confound detected across resolution clusters.")


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
    df_m_audit.to_csv(mendeley_path, index=False, encoding="utf-8-sig")
    df_k_audit.to_csv(kau_path,      index=False, encoding="utf-8-sig")
    print(f"\n  Audit CSVs saved:")
    print(f"    {mendeley_path}")
    print(f"    {kau_path}")

    # Export validated clean KAU paths for downstream stages (_2b_feature_pca.py, _10_11_external_val.py)
    clean_k = df_k_audit[~df_k_audit["is_corrupt"]].copy()
    valid_records = clean_k[["path", "label", "label_int", "width", "height"]].to_dict(orient="records")
    valid_json_path = OUT_DIR / "kau_valid_paths.json"
    with open(valid_json_path, "w") as f:
        json.dump(valid_records, f, indent=2)
    print(f"    {valid_json_path} ({len(valid_records)} verified clean images)")


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
    df_k_paths = collect_kau_paths(ROOT_KAU, KAU_BIRAD_MAP)
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
    assert_no_format_confound(df_k_audit, max_allowed_diff=0.25)

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