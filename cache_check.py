"""
cache_check.py
==============
Pluggable cache guard for the QML pipeline.
Paths are resolved dynamically relative to this file's location —
no hardcoded absolute paths, so it works on any machine (Kaggle, local, RTX).

Usage
-----
from cache_check import already_done, CACHE

# At the top of any script's main():
if already_done("eda"):
    return   # skip recomputation

# At the end of any script's main():
CACHE.mark_done("eda")

# To force a rerun of a stage:
from cache_check import CACHE
CACHE.reset("eda")      # reset one stage
CACHE.reset()           # reset all stages
"""

import json
from pathlib import Path

# ── Resolve output root dynamically ───────────────────────────────────────────
# cache_check.py lives in the same directory as all pipeline scripts.
# Outputs are written to an "outputs/" subdirectory next to the scripts,
# OR to the same directory if running on Kaggle (/kaggle/working/).
#
# Detection order:
#   1. PIPELINE_OUTPUT_DIR environment variable (explicit override — recommended)
#   2. /kaggle/working  (if running on Kaggle)
#   3. <script_dir>/outputs  (local machine default)

import os

_SCRIPT_DIR = Path(__file__).resolve().parent

if "PIPELINE_OUTPUT_DIR" in os.environ:
    _OUT = Path(os.environ["PIPELINE_OUTPUT_DIR"]).resolve()
elif Path("/kaggle/working").exists():
    _OUT = Path("/kaggle/working")
else:
    _OUT = _SCRIPT_DIR / "outputs"

# Ensure output root exists
_OUT.mkdir(parents=True, exist_ok=True)

# ── Central registry of required sentinel files per stage ─────────────────────
# A stage is "done" when ALL its sentinel files exist on disk.
# Edit only the filenames here — paths are built from _OUT automatically.

_SENTINELS = {
    "eda": [
        _OUT / "eda_outputs" / "mendeley_audit.csv",
        _OUT / "eda_outputs" / "kau_audit.csv",
        _OUT / "eda_outputs" / "mendeley_split_indices.json",
    ],
    "baseline": [
        _OUT / "baseline_outputs" / "mobilenetv2_best.pt",
        _OUT / "baseline_outputs" / "baseline_results.json",
        _OUT / "baseline_outputs" / "training_history.csv",
    ],
    "features": [
        _OUT / "feature_outputs" / "features_train_raw.npy",
        _OUT / "feature_outputs" / "features_kau_raw.npy",
        _OUT / "feature_outputs" / "pca_4_components.pkl",
        _OUT / "feature_outputs" / "pca_report.json",
    ],
    "vqc": [
        _OUT / "vqc_outputs" / "regime_A" / f"vqc_q4_l2_lr0.01.pt",
        _OUT / "vqc_outputs" / "ablation_table.csv",
    ],
    "uq": [
        _OUT / "uq_outputs" / "mc_dropout_uncertainty.csv",
        _OUT / "uq_outputs" / "quantum_shot_variance.csv",
        _OUT / "uq_outputs" / "uq_summary.json",
    ],
    "qfl": [
        _OUT / "qfl_outputs" / "federated_training.csv",
        _OUT / "qfl_outputs" / "qfl_summary.json",
    ],
    "external_validation": [
        _OUT / "external_val_outputs" / "final_ablation_table.csv",
        _OUT / "external_val_outputs" / "generalisation_report.json",
    ],
}


class _PipelineCache:
    """Tracks completed pipeline stages in a JSON registry file."""

    def __init__(self):
        self.registry_path = _OUT / ".pipeline_cache.json"
        self._completed: set = self._load()

    def _load(self) -> set:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return set(json.load(f).get("completed", []))
        return set()

    def _save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump({"completed": sorted(self._completed)}, f, indent=2)

    def mark_done(self, stage: str):
        self._completed.add(stage)
        self._save()
        print(f"  ✓ Stage '{stage}' marked complete in cache.")

    def reset(self, stage: str = None):
        """Force recompute: reset one stage, or all stages if stage=None."""
        if stage:
            self._completed.discard(stage)
            print(f"  Cache reset for stage: '{stage}'")
        else:
            self._completed.clear()
            print("  Full cache reset.")
        self._save()

    @property
    def output_root(self) -> Path:
        return _OUT


# Singleton — import CACHE anywhere in the pipeline to share state
CACHE = _PipelineCache()


def already_done(stage: str, force: bool = False) -> bool:
    """
    Returns True (skip recomputation) when:
      - All sentinel files for the stage exist on disk, AND
      - The stage is recorded in the cache registry.

    Parameters
    ----------
    stage : str
        Stage key — one of: eda, baseline, features, vqc, uq, qfl,
        external_validation. Add new stages to _SENTINELS above.
    force : bool
        If True, always returns False regardless of cache state.
        Use for debugging: already_done("eda", force=True)
    """
    if force:
        print(f"  [force=True] Recomputing stage '{stage}'.")
        return False

    if stage not in _SENTINELS:
        print(f"  [WARNING] Unknown stage '{stage}'. "
              f"Valid stages: {list(_SENTINELS.keys())}")
        return False

    missing = [str(p) for p in _SENTINELS[stage] if not Path(p).exists()]

    if missing:
        print(f"  Stage '{stage}': {len(missing)} sentinel file(s) missing "
              f"— will recompute.")
        for m in missing:
            print(f"    ✗ {m}")
        return False

    if stage not in CACHE._completed:
        # Files exist but registry wasn't written (e.g. previous session / Kaggle restart).
        # Trust the files and auto-register.
        print(f"  Stage '{stage}': all files present "
              f"(auto-registering from previous run).")
        CACHE.mark_done(stage)

    print(f"  Stage '{stage}': cached ✓ — skipping recomputation.")
    print(f"  Output root: {_OUT}")
    return True


def print_status():
    """Print the current cache status for all registered stages."""
    print(f"\n  Pipeline cache status  (output root: {_OUT})")
    print(f"  Registry: {CACHE.registry_path}")
    print("  " + "─" * 55)
    for stage, sentinels in _SENTINELS.items():
        all_exist = all(Path(p).exists() for p in sentinels)
        registered = stage in CACHE._completed
        status = (
            "✓ complete" if all_exist and registered else
            "⚡ files exist, not registered" if all_exist else
            "✗ missing files"
        )
        print(f"  {stage:25s} {status}")
    print()