"""Strict Train / Validation / Held-out Test Split and Leakage Verification.

Defines:
- Held-out test set (7 files strictly isolated):
    * Discharge_02.csv (0.5A)
    * Pair 10: Discharge_10.csv, Load_10.csv (1.0A)
    * Pair 20: Discharge_20.csv, Load_20.csv (2.0A)
    * Pair 30: Discharge_30.csv, Load_30.csv (3.0A)
- Validation set: stratified by whole file (never random rows):
    * Discharge: Discharge_05.csv, Discharge_15.csv, Discharge_25.csv
    * Charge: Load_5.csv, Load_15.csv, Load_25.csv
- Train set: remaining 47 files.
- Automated leakage tests guaranteeing zero test contamination in training or scaler.
"""

from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import numpy as np

from src.config import load_config


# Canonical split definitions
HELD_OUT_TEST_FILES: List[str] = [
    "Discharge_02.csv",
    "Discharge_10.csv",
    "Load_10.csv",
    "Discharge_20.csv",
    "Load_20.csv",
    "Discharge_30.csv",
    "Load_30.csv",
]

DEFAULT_VAL_FILES: List[str] = [
    "Discharge_05.csv",
    "Discharge_15.csv",
    "Discharge_25.csv",
    "Load_5.csv",
    "Load_15.csv",
    "Load_25.csv",
]


def get_dataset_splits(
    all_filenames: List[str],
    val_filenames: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Partition list of all 60 filenames into train, val, and held-out test sets."""
    if val_filenames is None:
        val_filenames = DEFAULT_VAL_FILES
        
    test_set = set(HELD_OUT_TEST_FILES)
    val_set = set(val_filenames)
    all_set = set(all_filenames)
    
    # Assert held-out test files exist in all_filenames
    missing_test = test_set - all_set
    if missing_test:
        raise ValueError(f"Held-out test files missing from dataset: {missing_test}")
        
    # Assert validation files exist in all_filenames
    missing_val = val_set - all_set
    if missing_val:
        raise ValueError(f"Validation files missing from dataset: {missing_val}")
        
    # Disjoint check
    overlap_test_val = test_set & val_set
    if overlap_test_val:
        raise ValueError(f"Overlap between test and val sets: {overlap_test_val}")
        
    train_set = all_set - test_set - val_set
    
    splits = {
        "train": sorted(list(train_set)),
        "val": sorted(list(val_set)),
        "test": sorted(list(test_set)),
    }
    
    # Verify leakage
    verify_no_leakage(splits)
    
    return splits


def verify_no_leakage(splits: Dict[str, List[str]]) -> bool:
    """Automated leakage gate: strictly asserts no contamination across splits."""
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    # 1. Test set must NOT overlap with train or val
    leak_train = train_set & test_set
    if leak_train:
        raise AssertionError(f"CRITICAL LEAKAGE: Test files in training set! {leak_train}")
        
    leak_val = val_set & test_set
    if leak_val:
        raise AssertionError(f"CRITICAL LEAKAGE: Test files in validation set! {leak_val}")
        
    # 2. Train and val must be disjoint
    leak_train_val = train_set & val_set
    if leak_train_val:
        raise AssertionError(f"CRITICAL LEAKAGE: Train and val sets overlap! {leak_train_val}")
        
    # 3. Exactly 7 held-out files
    if len(test_set) != 7:
        raise AssertionError(f"Test set must have exactly 7 files, found {len(test_set)}")
        
    # 4. Total files must be 60 if all are present
    total = len(train_set) + len(val_set) + len(test_set)
    if total != 60:
        raise AssertionError(f"Total files in splits must equal 60, found {total}")
        
    return True


def fit_scaler_on_train(train_dfs: List[pd.DataFrame], feature_cols: List[str] = ["V", "I", "V_smooth"]) -> Dict[str, np.ndarray]:
    """Fit feature standardizer strictly on training data only.
    
    Returns mean and std vectors for normalisation: x_norm = (x - mean) / std.
    """
    all_features = []
    for df in train_dfs:
        all_features.append(df[feature_cols].to_numpy(dtype=np.float64))
        
    stacked = np.vstack(all_features)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    
    # Prevent division by zero
    std = np.where(std < 1e-6, 1.0, std)
    
    return {
        "features": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


if __name__ == "__main__":
    from src.data_prep import load_cached_dataset
    print("=" * 70)
    print("TEST PROTOCOL & LEAKAGE VERIFICATION")
    print("=" * 70)
    dataset = load_cached_dataset()
    splits = get_dataset_splits(list(dataset.keys()))
    print(f"Train files ({len(splits['train'])}): {splits['train'][:5]} ...")
    print(f"Val files   ({len(splits['val'])}): {splits['val']}")
    print(f"Test files  ({len(splits['test'])}): {splits['test']}")
    print("Verification passed: Zero leakage detected!")
