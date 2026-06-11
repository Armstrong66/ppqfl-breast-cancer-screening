# uq_smoke_test.py

import re
import pandas as pd
from pathlib import Path

def auto_detect_best_vqc_config(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    all_ckpts = list(ckpt_dir.glob("vqc_q*.pt"))
    
    if not all_ckpts:
        print(f"? SMOKE TEST FAILED: No VQC checkpoints found in {ckpt_dir}")
        print("   Did 3_5_vqc.py finish saving files to regime_A?")
        return None

    best_auc = -1.0
    best_config = None

    print(f"Found {len(all_ckpts)} checkpoint file(s). Parsing configurations...")

    for ckpt_path in all_ckpts:
        match = re.search(r"vqc_q(\d+)_l(\d+)_lr([\d\.]+)\.pt", ckpt_path.name)
        if match:
            n_qubits = int(match.group(1))
            n_layers = int(match.group(2))
            lr = float(match.group(3))
            
            history_file = ckpt_dir / f"vqc_q{n_qubits}_l{n_layers}_lr{lr}_history.csv"
            status = "No history file"
            
            if history_file.exists():
                try:
                    history_df = pd.read_csv(history_file)
                    max_val_auc = history_df["val_auc"].max()
                    status = f"History found (Max Val AUC: {max_val_auc:.4f})"
                    if max_val_auc > best_auc:
                        best_auc = max_val_auc
                        best_config = (n_qubits, n_layers, lr)
                except Exception as e:
                    status = f"History error ({str(e)})"
            
            print(f"  - {ckpt_path.name} -> {status}")
            
            if best_config is None:
                best_config = (n_qubits, n_layers, lr)
                print(f"Fellback on default best_configs, not actual ones from the 3_5_vqc output file")

    return best_config, best_auc

# Run the test against your actual path
vqc_outputs_path = "/home/derrick/Projects/QFL_breast_cancer_screening/outputs/vqc_outputs/regime_A"
print("="*60)
print(" RUNNING VQC CONFIG AUTO-DETECT SMOKE TEST")
print("="*60)

result = auto_detect_best_vqc_config(vqc_outputs_path)

if result:
    config, auc = result
    print("="*60)
    print("?? SMOKE TEST PASSED SUCCESSFULLY!")
    print(f"   Best Config Picked: Qubits={config[0]}, Layers={config[1]}, LR={config[2]}")
    print(f"   Highest Validation AUC: {auc:.4f}")
    print("="*60)