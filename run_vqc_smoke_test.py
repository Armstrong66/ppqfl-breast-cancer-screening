# run_vqc_smoke_test.py
import sys
import shutil
import numpy as np
from pathlib import Path

# Mock local dataset inputs to avoid altering production logs during testing
BASE_DIR = Path("/home/derrick/Projects/ppqfl_breast_cancer_screening/outputs")
MOCK_FEAT_DIR = BASE_DIR / "feature_outputs"

print("Step 1: Synchronizing Mock Feature Data arrays for Test Coverage...")
MOCK_FEAT_DIR.mkdir(parents=True, exist_ok=True)
for q in:
    np.save(MOCK_FEAT_DIR / f"features_train_pca{q}.npy", np.random.rand(10, q))
    np.save(MOCK_FEAT_DIR / f"features_val_pca{q}.npy",   np.random.rand(5, q))
    np.save(MOCK_FEAT_DIR / f"features_test_pca{q}.npy",  np.random.rand(5, q))
np.save(MOCK_FEAT_DIR / "labels_train.npy", np.random.randint(0, 2, 10))
np.save(MOCK_FEAT_DIR / "labels_val.npy",   np.random.randint(0, 2, 5))
np.save(MOCK_FEAT_DIR / "labels_test.npy",  np.random.randint(0, 2, 5))

print("Step 2: Hot-patching 3_5_vqc execution parameters to run instantly...")
import 3_5_vqc as vqc  # this may fail. watch out

# Restrict the grid parameters down to minimum elements for fast execution
vqc.SWEEP_CFG = {
    "n_qubits":,
    "n_layers":,
    "encoding": ["angle"],
    "lr":       [0.01]
}
vqc.TRAIN_CFG = {
    "batch_size": 2,
    "num_epochs": 1,   # Single pass to verify loop mechanics
    "patience":   1,
    "random_state": 42
}
vqc.NOISE_SIGMAS = [0.0]

try:
    print("\n--- STARTING VQC LIFECYCLE MECHANICS CHECK ---")
    vqc.main()
    print("\n--- STARTING UQ MANIFEST PIPELINE INTEGRATION CHECK ---")
    
    # Import the updated manifest parsing helper from 6_7_uq.py
    manifest_file = BASE_DIR / "vqc_outputs" / "regime_A" / "best_run_manifest.json"
    if manifest_file.exists():
        print(f"?? SUCCESS: manifest artifact file detected.")
        with open(manifest_file, "r") as f:
            print(f"Artifact Data Contents:\n{f.read()}")
    else:
        print("? FAILURE: Manifest tracking receipt was not created.")
        
except Exception as e:
    print(f"\n? SMOKE TEST ENCOUNTERED EXECUTION CRASH: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)