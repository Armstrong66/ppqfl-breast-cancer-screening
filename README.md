# Privacy-Preserving Quantum Federated Learning for Breast Cancer Screening
### African and MENA Population Contexts

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.41-brightgreen.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Global Health Artificial Intelligence and Computing Laboratory - KCCR, KNUST, Ghana.

---

## Overview
This project implements a hybrid quantum-classical neural network (HQCNN)
within a simulated Quantum Federated Learning (QFL) framework for
privacy-preserving breast cancer screening. Three virtual Ghanaian hospital
clients (Accra, Kumasi, Tamale) train locally; only VQC parameters are
aggregated — raw patient data never leaves any node.

## Key contributions
- First QFL pipeline benchmarked on African mammography data
- HQCNN: MobileNetV2 feature extractor + Variational Quantum Circuit (9–25 params)
- Differential privacy simulation (σ_dp sweep) with privacy-utility trade-off analysis
- Uncertainty quantification: MC-Dropout + quantum shot variance
- Cross-population external validation: SA → MENA generalisation gap
- Temperature scaling for VQC calibration (ECE correction)

---

## Repository Structure

```
ppqfl-breast-cancer-screening/
│
├── _1_eda.py                   # Phase 1: Data audit & EDA (Mendeley + KAU-BCMD)
├── _2a_baseline.py             # Phase 2a: MobileNetV2 classical baseline
├── _2b_feature_pca.py          # Phase 2b: Feature extraction + PCA → quantum bridge
├── _3_5_vqc.py                 # Phases 3–5: VQC design, Regime A/B, sweep, noise
├── _6_7_uq.py                  # Phases 6–7: Uncertainty quantification + temperature scaling
├── _8_9_qfl.py                 # Phases 8–9: Simulated QFL + differential privacy
├── _10_11_external_val.py      # Phases 10–11: KAU external validation + ablation table
│
├── cache_check.py              # Pipeline cache guard (skip completed stages)
├── run_pipeline.sh             # Full pipeline orchestrator (nohup / screen ready)
├── setup_local.sh              # Local machine setup (path migration + deps)
│
├── requirements.txt            # Pinned dependencies
├── .gitignore
└── README.md
```

**Outputs**:
```
outputs/
├── eda_outputs/
├── baseline_outputs/
├── feature_outputs/
├── vqc_outputs/  (regime_A/, regime_B/, sweep/, noise/)
├── uq_outputs/
├── qfl_outputs/
├── external_val_outputs/
└── pipeline_logs/
```

---

## Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/Armstrong66/ppqfl-breast-cancer-screening.git
cd ppqfl-breast-cancer-screening

# If using conda (recommended):
conda activate qfl_breast_cancer
python3 -m pip install -r requirements.txt

# Or run the full local setup script:
chmod +x setup_local.sh
./setup_local.sh --data-dir /path/to/your/datasets
```

### 2. Add datasets

| Dataset | Source | Place at |
|---------|--------|----------|
| Mendeley Mammogram (Polokwane, SA) | [Mendeley Data](https://doi.org/10.17632/88vzgys5vg.2) | `data/mendeley/Breast Cancer Original/` |
| KAU-BCMD (Saudi Arabia) | [Kaggle CC0](https://www.kaggle.com/asmaasaad/king-abdulaziz-university-mammogram-dataset) | `data/kau/` |

Update `BASE_DATA_DIR` in each script if your data lives elsewhere.

### 3. Run

```bash
# Full pipeline (overnight — use nohup or screen)
mkdir -p pipeline_logs
nohup bash run_pipeline.sh > pipeline_logs/nohup_full.log 2>&1 &
echo "PID: $!"

# Monitor
tail -f pipeline_logs/nohup_full.log

# Resume from a specific stage after interruption
bash run_pipeline.sh --from 6_7_uq

# Run one stage only
bash run_pipeline.sh --only _8_9_qfl
```

### 4. Reset cache (force rerun of a stage)

```python
from cache_check import CACHE
CACHE.reset("vqc")   # reset one stage
CACHE.reset()        # reset all
```

---

## Pipeline Phases

| Phase | Script | Key output |
|-------|--------|------------|
| 1 | `_1_eda.py` | EDA report, stratified split indices |
| 2a | `_2a_baseline.py` | MobileNetV2 checkpoint, classical AUC |
| 2b | `_2b_feature_pca.py` | PCA-compressed `.npy` feature arrays |
| 3–5 | `_3_5_vqc.py` | HQCNN Regime A/B results, ablation table |
| 6–7 | `_6_7_uq.py` | MC-Dropout + quantum shot variance, ECE, temperature scaling |
| 8–9 | `_8_9_qfl.py` | QFL simulation, privacy-utility trade-off curve |
| 10–11 | `_10_11_external_val.py` | KAU cross-population results, master ablation table |

---

## Key Results (summary)

| Model | Params | Mendeley AUC | Notes |
|-------|--------|-------------|-------|
| MobileNetV2 (classical) | 164,226 | 0.9858 | Progressive unfreeze |
| HQCNN Regime A (q=4, l=2) | **9** | 0.9631 | Frozen backbone + VQC |
| HQCNN Regime B (q=4, l=2) | 29 | 0.9827 | End-to-end projection + VQC |
| QFL (3 clients, no DP) | 9 | ~0.96 | Simulated Ghanaian hospitals |

Classical ECE: 0.047 · Quantum ECE (raw): 0.223 · Quantum ECE (post-scaling): see `uq_outputs/uq_summary.json`

---

## Citation

If you use this code, please cite:

```bibtex
@misc{derrick2026qfl,
  title   = {Privacy-Preserving Quantum Federated Learning for Breast Cancer
             Screening in African and MENA Populations},
  author  = {A. N. K, Joseph Derrick},
  year    = {2026},
  note    = {Global Health Artificial Intelligence and Computing Laboratory - KCCR, KNUST, Ghana.},
  url     = {https://github.com/Armstrong66/ppqfl-breast-cancer-screening}
}
```

---

## License

MIT — see [LICENSE](LICENSE).  
Datasets are subject to their own licences (Mendeley CC BY 4.0; KAU-BCMD CC0).