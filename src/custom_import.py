"""Custom Data Import, Column Mapping, Preprocessing, and ML SOC Prediction.

Supports:
- Predefined default columns: Time, Voltage, Current (optional: Temperature, SOC_True).
- Arbitrary values inside: any voltage range, any current profile (discharge/charge/rest/pulse),
  any sampling duration, and any number of samples.
- Auto-detection of delimiters (comma, semicolon, tab).
- Auto-mapping of column aliases (e.g. V, Vbat, Vbat1..4, I, Amps, t, Time_s).
- Causal feature engineering: V_smooth via strictly causal moving average (12-sample expanding window).
- Vectorized PureNumpyMLP forward pass predicting SOC in [0, 100]%.
- Coulomb Counting benchmark integration.
- Export of predictions to CSV.
"""

import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

from src.config import get_path
from src.data_prep import compute_causal_moving_average
from src.mlp_infer import PureNumpyMLP


# Predefined default column specifications
PREDEFINED_DEFAULT_COLUMNS = ["Time", "Voltage", "Current"]
OPTIONAL_COLUMNS = ["Temperature", "SOC_True"]

COLUMN_ALIASES = {
    "voltage": ["voltage", "v", "vbat", "v_cell", "cell_voltage", "volt", "volts"],
    "current": ["current", "i", "amps", "curr", "amp", "current (a)"],
    "time": ["time", "time_s", "time (s)", "t", "timestamp", "seconds", "sec"],
    "temperature": ["temperature", "temp", "temperature_c", "temp_c", "degc", "t_c"],
    "soc": ["soc", "soc_true", "true_soc", "soc (%)", "state_of_charge", "actual_soc"],
}


def generate_sample_csv_template(n_samples: int = 120, dt_s: float = 5.0) -> str:
    """Generate a clean CSV string with predefined default columns and realistic arbitrary Li-ion profile.
    
    Default columns: Time, Voltage, Current, Temperature
    """
    t = np.arange(n_samples) * dt_s
    # Arbitrary realistic discharge profile: 4.18V down to 3.25V with a 1.5A load pulse
    progress = np.linspace(0.0, 1.0, n_samples)
    # Non-linear OCV-like curve with IR drop
    v_base = 4.18 - 0.70 * progress - 0.23 * (progress ** 3)
    # Add minor noise
    noise = np.sin(progress * 20.0) * 0.005
    v = np.round(v_base + noise, 3)
    # Current: 1.5 A constant discharge with a rest period near the middle
    i = np.where((progress > 0.45) & (progress < 0.55), 0.0, 1.5)
    temp = np.round(25.0 + 3.5 * progress, 1)
    
    # Ground truth reference SOC (%) for evaluation
    soc_ref = np.round(np.clip(100.0 * (1.0 - progress), 0.0, 100.0), 1)

    df_sample = pd.DataFrame({
        "Time": t,
        "Voltage": v,
        "Current": i,
        "Temperature": temp,
        "SOC_True": soc_ref,
    })
    
    buf = io.StringIO()
    df_sample.to_csv(buf, index=False)
    return buf.getvalue()


def parse_and_predict_custom_data(
    content: str,
    filename: str = "custom_data.csv",
    model: Optional[PureNumpyMLP] = None,
) -> Dict[str, Any]:
    """Parse custom battery data with predefined columns and arbitrary values, and predict SOC.
    
    Parameters
    ----------
    content : str
        CSV, TSV, or semicolon-delimited text content.
    filename : str
        Source filename for identification.
    model : PureNumpyMLP, optional
        Preloaded model instance. If None, initialized from weights.
        
    Returns
    -------
    Dict containing:
        - status: "success"
        - filename: str
        - stats: summary metrics dictionary
        - series: time-series arrays for charting
        - preview: first 10 and last 10 rows
        - df_processed: DataFrame ready for live simulation replay
    """
    if not content or not content.strip():
        raise ValueError("Uploaded file content is empty.")

    # Read CSV with automatic delimiter sniffing (supports comma, semicolon, tab)
    try:
        df_raw = pd.read_csv(io.StringIO(content), sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV data: {str(e)}")

    if df_raw.empty:
        raise ValueError("Parsed dataset contains 0 rows.")

    col_map = {str(c).strip().lower(): c for c in df_raw.columns}

    # 1. Resolve Voltage Column (Predefined Default: 'Voltage' or 'V')
    v_col = None
    # Check for NASA Kelvin probe columns (Vbat1..4)
    vbat_cols = [
        col_map[k] for k in col_map
        if k.startswith("vbat") and any(k.endswith(str(d)) for d in range(1, 10))
    ]
    if vbat_cols:
        v_arr = df_raw[vbat_cols].mean(axis=1).to_numpy(dtype=float)
        v_col_name = "Vbat (mean of Kelvin probes)"
    else:
        for candidate in COLUMN_ALIASES["voltage"]:
            if candidate in col_map:
                v_col = col_map[candidate]
                break
        if v_col is None:
            # Fuzzy match
            for k, orig in col_map.items():
                if "volt" in k or k.startswith("v_") or k.endswith("_v"):
                    v_col = orig
                    break
        if v_col is None:
            raise ValueError(
                f"Missing required predefined 'Voltage' column. Found columns: {list(df_raw.columns)}. "
                f"Expected one of: {PREDEFINED_DEFAULT_COLUMNS}"
            )
        v_col_name = v_col
        v_arr = pd.to_numeric(df_raw[v_col], errors="coerce").to_numpy(dtype=float)

    # 2. Resolve Current Column (Predefined Default: 'Current' or 'I')
    i_col = None
    for candidate in COLUMN_ALIASES["current"]:
        if candidate in col_map:
            i_col = col_map[candidate]
            break
    if i_col is not None:
        i_col_name = i_col
        i_arr = pd.to_numeric(df_raw[i_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        i_col_name = "Current (defaulted to 0.0 A)"
        i_arr = np.zeros(len(df_raw), dtype=float)

    # 3. Resolve Time Column (Predefined Default: 'Time' or 'time_s')
    t_col = None
    for candidate in COLUMN_ALIASES["time"]:
        if candidate in col_map:
            t_col = col_map[candidate]
            break
    if t_col is not None:
        t_col_name = t_col
        t_arr = pd.to_numeric(df_raw[t_col], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isnan(t_arr)) or len(t_arr) == 0:
            t_arr = np.arange(len(df_raw), dtype=float) * 5.0
    else:
        t_col_name = "Time (generated 5.0s uniform grid)"
        t_arr = np.arange(len(df_raw), dtype=float) * 5.0

    # 4. Resolve Optional Temperature
    temp_col = None
    for candidate in COLUMN_ALIASES["temperature"]:
        if candidate in col_map and candidate != t_col:
            temp_col = col_map[candidate]
            break
    if temp_col is not None:
        temp_arr = pd.to_numeric(df_raw[temp_col], errors="coerce").fillna(25.0).to_numpy(dtype=float)
    else:
        temp_arr = np.full(len(df_raw), 25.0, dtype=float)

    # 5. Resolve Optional Ground Truth SOC
    soc_col = None
    for candidate in COLUMN_ALIASES["soc"]:
        if candidate in col_map:
            soc_col = col_map[candidate]
            break
    has_ground_truth = False
    soc_true_arr = None
    if soc_col is not None:
        parsed_soc = pd.to_numeric(df_raw[soc_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isnan(parsed_soc)):
            has_ground_truth = True
            soc_true_arr = parsed_soc

    # Filter out NaNs in voltage if any
    valid_mask = ~np.isnan(v_arr)
    if not np.all(valid_mask):
        t_arr = t_arr[valid_mask]
        v_arr = v_arr[valid_mask]
        i_arr = i_arr[valid_mask]
        temp_arr = temp_arr[valid_mask]
        if soc_true_arr is not None:
            soc_true_arr = soc_true_arr[valid_mask]

    n_samples = len(v_arr)
    if n_samples == 0:
        raise ValueError("No valid numeric voltage samples found in dataset.")

    # Calculate sampling dt (median delta)
    if n_samples > 1:
        diffs = np.diff(t_arr)
        positive_diffs = diffs[diffs > 0]
        dt_s = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 5.0
    else:
        dt_s = 5.0

    # Compute Feature 3: Strictly causal moving average (12-sample expanding window)
    v_smooth = compute_causal_moving_average(v_arr, window_samples=12)

    # Load model if not provided
    if model is None:
        model = PureNumpyMLP()

    # Predict SOC using PureNumpyMLP
    X = np.column_stack([v_arr, i_arr, v_smooth])
    soc_ml = model.predict_batch(X)
    soc_ml = np.clip(soc_ml, 0.0, 100.0)

    # Coulomb Counting comparison baseline
    # Initial SOC: True SOC if given, else first ML prediction
    initial_soc = float(soc_true_arr[0]) if has_ground_truth and not np.isnan(soc_true_arr[0]) else float(soc_ml[0])
    # Estimate total capacity or use nominal 2.6 Ah (9360 As)
    total_q_as = np.sum(0.5 * (np.abs(i_arr[:-1]) + np.abs(i_arr[1:])) * dt_s) if n_samples > 1 else 0.0
    nominal_cap_as = max(total_q_as, 2.6 * 3600.0)
    
    soc_cc = np.zeros(n_samples, dtype=float)
    soc_cc[0] = initial_soc
    for k in range(1, n_samples):
        dq = i_arr[k] * dt_s
        soc_cc[k] = np.clip(soc_cc[k - 1] - (dq / nominal_cap_as) * 100.0, 0.0, 100.0)

    # Compute Error Metrics if ground truth is present
    ml_mae = None
    ml_rmse = None
    ml_max_error = None
    if has_ground_truth and soc_true_arr is not None:
        valid_gt = ~np.isnan(soc_true_arr)
        if np.any(valid_gt):
            errs = np.abs(soc_true_arr[valid_gt] - soc_ml[valid_gt])
            ml_mae = round(float(np.mean(errs)), 2)
            ml_rmse = round(float(np.sqrt(np.mean(errs ** 2))), 2)
            ml_max_error = round(float(np.max(errs)), 2)

    # Build DataFrame for live twin replay and export
    df_processed = pd.DataFrame({
        "time_s": t_arr,
        "V": v_arr,
        "I": i_arr,
        "V_smooth": v_smooth,
        "temperature_c": temp_arr,
        "soc_ml": soc_ml,
        "soc_true": soc_true_arr if has_ground_truth else soc_ml,
        "soc_cc": soc_cc,
    })

    # Summary Statistics
    duration_s = float(t_arr[-1] - t_arr[0]) if n_samples > 1 else 0.0
    duration_min = round(duration_s / 60.0, 1)
    
    stats = {
        "sample_count": int(n_samples),
        "duration_s": round(duration_s, 1),
        "duration_min": duration_min,
        "dt_s": round(dt_s, 2),
        "v_min": round(float(np.min(v_arr)), 3),
        "v_max": round(float(np.max(v_arr)), 3),
        "v_mean": round(float(np.mean(v_arr)), 3),
        "i_min": round(float(np.min(i_arr)), 3),
        "i_max": round(float(np.max(i_arr)), 3),
        "i_mean": round(float(np.mean(i_arr)), 3),
        "soc_start": round(float(soc_ml[0]), 1),
        "soc_end": round(float(soc_ml[-1]), 1),
        "soc_min": round(float(np.min(soc_ml)), 1),
        "soc_max": round(float(np.max(soc_ml)), 1),
        "soc_delta": round(float(abs(soc_ml[0] - soc_ml[-1])), 1),
        "has_ground_truth": has_ground_truth,
        "ml_mae": ml_mae,
        "ml_rmse": ml_rmse,
        "ml_max_error": ml_max_error,
        "column_detected_v": v_col_name,
        "column_detected_i": i_col_name,
        "column_detected_t": t_col_name,
    }

    # Generate Chart Series (downsample to at most 1,500 points for crisp web rendering)
    stride = max(1, n_samples // 1500)
    indices = np.arange(0, n_samples, stride)
    if indices[-1] != n_samples - 1:
        indices = np.append(indices, n_samples - 1)

    chart_series = {
        "time": [round(float(t_arr[idx]), 1) for idx in indices],
        "voltage": [round(float(v_arr[idx]), 3) for idx in indices],
        "v_smooth": [round(float(v_smooth[idx]), 3) for idx in indices],
        "current": [round(float(i_arr[idx]), 3) for idx in indices],
        "predicted_soc": [round(float(soc_ml[idx]), 2) for idx in indices],
        "coulomb_soc": [round(float(soc_cc[idx]), 2) for idx in indices],
        "true_soc": [round(float(soc_true_arr[idx]), 2) if has_ground_truth and not np.isnan(soc_true_arr[idx]) else None for idx in indices],
    }

    # Generate preview table rows (first 5 and last 5)
    preview_indices = list(range(min(5, n_samples)))
    if n_samples > 5:
        tail_start = max(5, n_samples - 5)
        for idx in range(tail_start, n_samples):
            if idx not in preview_indices:
                preview_indices.append(idx)

    preview_rows = []
    for idx in preview_indices:
        preview_rows.append({
            "row": idx + 1,
            "time_s": round(float(t_arr[idx]), 1),
            "voltage": round(float(v_arr[idx]), 3),
            "current": round(float(i_arr[idx]), 3),
            "v_smooth": round(float(v_smooth[idx]), 3),
            "predicted_soc": round(float(soc_ml[idx]), 2),
            "true_soc": round(float(soc_true_arr[idx]), 2) if has_ground_truth and not np.isnan(soc_true_arr[idx]) else "N/A",
        })

    return {
        "status": "success",
        "filename": filename,
        "stats": stats,
        "series": chart_series,
        "preview": preview_rows,
        "df_processed": df_processed,
    }


def export_predictions_csv(df_processed: pd.DataFrame) -> str:
    """Export processed dataframe with predictions as CSV string."""
    export_df = pd.DataFrame({
        "Time_s": df_processed["time_s"],
        "Voltage_V": df_processed["V"],
        "Current_A": df_processed["I"],
        "V_smooth_V": df_processed["V_smooth"],
        "Temperature_C": df_processed["temperature_c"],
        "Predicted_SOC_percent": df_processed["soc_ml"],
        "CoulombCounting_SOC_percent": df_processed["soc_cc"],
    })
    if "soc_true" in df_processed and not np.all(df_processed["soc_true"] == df_processed["soc_ml"]):
        export_df["True_SOC_percent"] = df_processed["soc_true"]
        export_df["Absolute_Error_percent"] = np.abs(df_processed["soc_true"] - df_processed["soc_ml"])

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    return buf.getvalue()
