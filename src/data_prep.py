"""Data Preprocessing, Resampling, Causal Feature Engineering, and Caching.

Implements:
1. V = mean(Vbat1..Vbat4)
2. Uniform 5s grid resampling (linear interpolation default, bin-average option)
3. Discharge-positive current convention (charge current inverted to negative)
4. Monotonic SOC labels [0, 100]% via normalized coulomb counting per file
5. Strictly causal 60s moving average (12 samples @ 5s) with expanding warm-up
6. Parquet/npz dataset caching with integrity hash
7. Pair rebuilding utility for twin simulation and evaluation
"""

import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.config import get_path, load_config


def extract_file_index(filename: str) -> int:
    """Extract numeric index from filename."""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1


def compute_causal_moving_average(v_arr: np.ndarray, window_samples: int = 12) -> np.ndarray:
    """Compute strictly causal moving average with expanding window warm-up.
    
    At index k < window_samples: mean(v_arr[0 : k+1])
    At index k >= window_samples: mean(v_arr[k - window_samples + 1 : k+1])
    Guarantees zero future leakage and matches firmware ring buffer.
    """
    n = len(v_arr)
    v_smooth = np.zeros(n, dtype=np.float64)
    # Cumulative sum for exact O(1) sliding window computation
    cumsum = np.cumsum(v_arr, dtype=np.float64)
    
    # Warm-up phase: expanding window
    for k in range(min(window_samples, n)):
        v_smooth[k] = cumsum[k] / (k + 1)
        
    # Steady state phase: fixed window size
    if n > window_samples:
        window_sums = cumsum[window_samples:] - cumsum[:-window_samples]
        v_smooth[window_samples:] = window_sums / window_samples
        
    return v_smooth


def resample_file_data(
    df: pd.DataFrame,
    file_type: str,
    sampling_interval_s: float = 5.0,
    resample_method: str = "linear",
) -> pd.DataFrame:
    """Resample raw battery cycling dataframe to uniform time grid with causal features and SOC labels."""
    time_raw = df["Time"].to_numpy(dtype=np.float64)
    t_min = time_raw[0]
    t_max = time_raw[-1]
    
    # Uniform time grid starting at t=0
    t_grid = np.arange(0.0, (t_max - t_min), sampling_interval_s, dtype=np.float64)
    time_rel = time_raw - t_min
    
    # Raw signals
    v_raw = df[["Vbat1", "Vbat2", "Vbat3", "Vbat4"]].mean(axis=1).to_numpy(dtype=np.float64)
    i_raw = df["Current"].to_numpy(dtype=np.float64)
    temp_raw = df["Temperature"].to_numpy(dtype=np.float64)
    
    if resample_method == "linear":
        v_resampled = np.interp(t_grid, time_rel, v_raw)
        i_resampled = np.interp(t_grid, time_rel, i_raw)
        temp_resampled = np.interp(t_grid, time_rel, temp_raw)
    elif resample_method == "bin_average":
        # Bin-averaging into 5s windows
        bin_indices = np.digitize(time_rel, t_grid) - 1
        bin_indices = np.clip(bin_indices, 0, len(t_grid) - 1)
        v_resampled = np.zeros(len(t_grid), dtype=np.float64)
        i_resampled = np.zeros(len(t_grid), dtype=np.float64)
        temp_resampled = np.zeros(len(t_grid), dtype=np.float64)
        for b in range(len(t_grid)):
            mask = (bin_indices == b)
            if np.any(mask):
                v_resampled[b] = np.mean(v_raw[mask])
                i_resampled[b] = np.mean(i_raw[mask])
                temp_resampled[b] = np.mean(temp_raw[mask])
            else:
                v_resampled[b] = np.interp(t_grid[b], time_rel, v_raw)
                i_resampled[b] = np.interp(t_grid[b], time_rel, i_raw)
                temp_resampled[b] = np.interp(t_grid[b], time_rel, temp_raw)
    else:
        raise ValueError(f"Unknown resample method: {resample_method}")
        
    # Current sign convention: DISCHARGE POSITIVE
    # In charge files, raw current is positive; flip so charging is negative
    if file_type == "charge":
        i_signed = -np.abs(i_resampled)
    else:
        i_signed = np.abs(i_resampled)
        
    # Compute normalized coulomb counting SOC labels
    # Use trapezoidal integration over uniform 5s grid
    dt = sampling_interval_s
    abs_curr = np.abs(i_signed)
    # Trapezoidal rule for cumulative capacity
    q_increments = 0.5 * (abs_curr[:-1] + abs_curr[1:]) * dt
    q_cum = np.concatenate([[0.0], np.cumsum(q_increments)])
    measured_capacity_as = q_cum[-1] # Ampere-seconds
    
    if measured_capacity_as <= 0.0:
        raise ValueError("Measured capacity is non-positive!")
        
    if file_type == "discharge":
        soc = 100.0 * (1.0 - (q_cum / measured_capacity_as))
    else:
        # Charge: 0% to 100%
        soc = 100.0 * (q_cum / measured_capacity_as)
        
    soc = np.clip(soc, 0.0, 100.0)
    
    # Assert monotonicity
    if file_type == "discharge":
        assert np.all(np.diff(soc) <= 1e-7), f"Discharge SOC not monotonic non-increasing!"
        assert abs(soc[0] - 100.0) < 1e-4, f"Discharge SOC does not start at 100%: {soc[0]}"
        assert abs(soc[-1] - 0.0) < 1e-4, f"Discharge SOC does not end at 0%: {soc[-1]}"
    else:
        assert np.all(np.diff(soc) >= -1e-7), f"Charge SOC not monotonic non-decreasing!"
        assert abs(soc[0] - 0.0) < 1e-4, f"Charge SOC does not start at 0%: {soc[0]}"
        assert abs(soc[-1] - 100.0) < 1e-4, f"Charge SOC does not end at 100%: {soc[-1]}"
        
    # Feature 3: Strictly Causal 60s moving average (12 samples @ 5s)
    v_smooth = compute_causal_moving_average(v_resampled, window_samples=12)
    
    res_df = pd.DataFrame({
        "time_s": t_grid,
        "V": v_resampled,
        "I": i_signed,
        "V_smooth": v_smooth,
        "temperature_c": temp_resampled,
        "soc_true": soc,
    })
    
    return res_df


def process_all_files(resample_method: str = "linear") -> Dict[str, pd.DataFrame]:
    """Process all 60 cycling files and return dictionary keyed by filename."""
    raw_dir = get_path("raw_dir")
    dis_dir = raw_dir / "Discharge_folder"
    chg_dir = raw_dir / "Load_folder"
    
    processed_dict = {}
    
    for f in sorted(list(dis_dir.glob("*.csv")), key=lambda p: extract_file_index(p.name)):
        df = pd.read_csv(f, sep=";")
        processed_dict[f.name] = resample_file_data(df, "discharge", 5.0, resample_method)
        
    for f in sorted(list(chg_dir.glob("*.csv")), key=lambda p: extract_file_index(p.name)):
        df = pd.read_csv(f, sep=";")
        processed_dict[f.name] = resample_file_data(df, "charge", 5.0, resample_method)
        
    return processed_dict


def build_cycle_pair(
    discharge_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    rest_duration_s: float = 300.0,
    sampling_interval_s: float = 5.0,
) -> pd.DataFrame:
    """Concatenate a discharge run and a charge run with an optional rest gap for full-cycle replay."""
    d_df = discharge_df.copy()
    c_df = charge_df.copy()
    
    t_offset_rest = d_df["time_s"].iloc[-1] + sampling_interval_s
    rest_steps = int(rest_duration_s / sampling_interval_s)
    
    if rest_steps > 0:
        t_rest = t_offset_rest + np.arange(rest_steps) * sampling_interval_s
        v_rest_start = d_df["V"].iloc[-1]
        v_rest_end = c_df["V"].iloc[0]
        # Smooth transition relaxation
        v_rest = np.linspace(v_rest_start, v_rest_end, rest_steps)
        rest_df = pd.DataFrame({
            "time_s": t_rest,
            "V": v_rest,
            "I": np.zeros(rest_steps),
            "V_smooth": compute_causal_moving_average(v_rest, 12),
            "temperature_c": np.full(rest_steps, d_df["temperature_c"].iloc[-1]),
            "soc_true": np.zeros(rest_steps),
        })
        t_offset_charge = t_rest[-1] + sampling_interval_s
        c_df["time_s"] = c_df["time_s"] + t_offset_charge
        combined = pd.concat([d_df, rest_df, c_df], ignore_index=True)
    else:
        c_df["time_s"] = c_df["time_s"] + t_offset_rest
        combined = pd.concat([d_df, c_df], ignore_index=True)
        
    # Recompute overall causal moving average continuously across the concatenated pair
    combined["V_smooth"] = compute_causal_moving_average(combined["V"].to_numpy(), 12)
    return combined


def cache_processed_dataset(processed_dict: Dict[str, pd.DataFrame], cache_dir: Path) -> str:
    """Cache all processed dataframes to parquet with metadata integrity hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hasher = hashlib.sha256()
    for name, df in sorted(processed_dict.items()):
        file_path = cache_dir / f"{name.replace('.csv', '')}.parquet"
        df.to_parquet(file_path, index=False)
        hasher.update(name.encode("utf-8"))
        hasher.update(df.to_numpy().tobytes())
        
    dataset_hash = hasher.hexdigest()
    meta = {
        "dataset_hash": dataset_hash,
        "num_files": len(processed_dict),
        "total_samples": int(sum(len(df) for df in processed_dict.values())),
        "files": sorted(list(processed_dict.keys())),
    }
    with open(cache_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[Prep] Cached {len(processed_dict)} files ({meta['total_samples']:,} total samples). Hash: {dataset_hash[:12]}")
    return dataset_hash


def load_cached_dataset(cache_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load pre-processed datasets from parquet cache."""
    if cache_dir is None:
        cache_dir = get_path("processed_dir")
    parquet_files = list(cache_dir.glob("*.parquet"))
    if not parquet_files:
        print("[Prep] Cache empty, processing raw files...")
        processed_dict = process_all_files()
        cache_processed_dataset(processed_dict, cache_dir)
        return processed_dict
        
    loaded = {}
    for p in parquet_files:
        name = f"{p.stem}.csv"
        loaded[name] = pd.read_parquet(p)
    return loaded


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2: PREPROCESSING AND FEATURE CACHING")
    print("=" * 70)
    processed = process_all_files()
    processed_dir = get_path("processed_dir")
    cache_processed_dataset(processed, processed_dir)
    print("Preprocessing completed successfully!")
