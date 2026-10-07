"""Unit tests for Phase 2: Data Preprocessing, Features, Labels, and Splits."""

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    compute_causal_moving_average,
    resample_file_data,
    process_all_files,
    build_cycle_pair,
)
from src.split import (
    get_dataset_splits,
    verify_no_leakage,
    fit_scaler_on_train,
    HELD_OUT_TEST_FILES,
)


def test_causal_moving_average_zero_future_leakage():
    """Verify that future samples have zero effect on earlier smoothed outputs."""
    v1 = np.array([3.8, 3.82, 3.84, 3.85, 3.86, 3.88, 3.90, 3.91, 3.92, 3.93, 3.94, 3.95, 3.96, 3.97])
    # Mutate the last 2 elements drastically
    v2 = v1.copy()
    v2[-2:] = [4.2, 4.3]
    
    s1 = compute_causal_moving_average(v1, window_samples=12)
    s2 = compute_causal_moving_average(v2, window_samples=12)
    
    # All elements before the mutation index must be identical
    np.testing.assert_allclose(s1[:-2], s2[:-2], atol=1e-12)


def test_causal_moving_average_warmup_matches_ring_buffer():
    """Verify expanding warm-up matches embedded ring buffer logic."""
    v = np.array([3.5, 3.6, 3.7, 3.8, 3.9])
    smooth = compute_causal_moving_average(v, window_samples=12)
    
    expected = [
        3.5,
        (3.5 + 3.6) / 2.0,
        (3.5 + 3.6 + 3.7) / 3.0,
        (3.5 + 3.6 + 3.7 + 3.8) / 4.0,
        (3.5 + 3.6 + 3.7 + 3.8 + 3.9) / 5.0,
    ]
    np.testing.assert_allclose(smooth, expected, atol=1e-12)


def test_soc_labels_bounds_and_monotonicity():
    """Verify synthetic and processed file SOC labels are strictly bounded and monotonic."""
    time = np.arange(0, 1000, 2.0)
    current = np.full_like(time, 1.0)
    v1 = np.linspace(4.1, 3.0, len(time))
    df_dis = pd.DataFrame({
        "Time": time,
        "Current": current,
        "Temperature": np.full_like(time, 25.0),
        "Vbat1": v1, "Vbat2": v1, "Vbat3": v1, "Vbat4": v1
    })
    
    res = resample_file_data(df_dis, "discharge", sampling_interval_s=5.0)
    soc = res["soc_true"].to_numpy()
    
    assert soc[0] == pytest.approx(100.0, abs=1e-3)
    assert soc[-1] == pytest.approx(0.0, abs=1e-3)
    assert np.all(soc >= 0.0) and np.all(soc <= 100.0)
    # Monotonic non-increasing
    assert np.all(np.diff(soc) <= 1e-6)
    # Time step is exactly 5.0s
    np.testing.assert_allclose(np.diff(res["time_s"].to_numpy()), 5.0)


def test_leakage_gate_enforcement():
    """Verify that verify_no_leakage raises AssertionError if test file is in train or val."""
    all_files = [f"file_{i}.csv" for i in range(60)]
    # Intentionally insert a held-out test file into train
    splits_leaked = {
        "train": ["Discharge_02.csv", "file_1.csv"],
        "val": ["file_2.csv"],
        "test": ["Discharge_02.csv", "Discharge_10.csv"],
    }
    with pytest.raises(AssertionError):
        verify_no_leakage(splits_leaked)


def test_scaler_fit_strictly_on_train_only():
    """Verify scaler computes statistics only on provided training dataframes."""
    df1 = pd.DataFrame({"V": [3.5, 4.0], "I": [1.0, 2.0], "V_smooth": [3.5, 3.75]})
    df2 = pd.DataFrame({"V": [3.6, 4.1], "I": [1.0, 2.0], "V_smooth": [3.6, 3.85]})
    
    scaler = fit_scaler_on_train([df1, df2])
    assert "mean" in scaler and "std" in scaler
    assert len(scaler["mean"]) == 3
    assert len(scaler["std"]) == 3
    # Check mean of V
    expected_mean_v = np.mean([3.5, 4.0, 3.6, 4.1])
    assert scaler["mean"][0] == pytest.approx(expected_mean_v, abs=1e-5)
