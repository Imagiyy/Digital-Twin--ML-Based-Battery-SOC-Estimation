"""Held-out Evaluation Pipeline (Phase 3).

Evaluates the trained MLP 3-16-1 against Coulomb Counting on the 7 held-out test files:
- Discharge 02 (0.5 A discharge only)
- Pair 10 (Discharge 10 + Load 10 @ 1.0 A)
- Pair 20 (Discharge 20 + Load 20 @ 2.0 A)
- Pair 30 (Discharge 30 + Load 30 @ 3.0 A)

ALL evaluations are conducted WITH realistic hardware noise:
- 5 mV ADC noise on pin
- 30 mA current sensor offset
- 12-bit ADC quantisation

Computes MAE, RMSE, Max Error, Discharging/Charging split, and outputs
markdown and CSV evaluation tables and high-res comparison figures.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import get_path, load_config
from src.data_prep import load_cached_dataset, build_cycle_pair
from src.mlp_infer import PureNumpyMLP
from src.train import apply_hardware_noise_augmentation


def simulate_coulomb_counting(
    i_measured: np.ndarray,
    dt_s: float,
    true_initial_soc: float,
    capacity_as: float,
) -> np.ndarray:
    """Simulate coulomb counter integrating the biased sensor starting from true initial SOC.
    
    Discharge positive convention:
    SOC(t) = SOC(0) - (integral(I dt) / capacity_as) * 100
    """
    q_integrated = np.concatenate([[0.0], np.cumsum(0.5 * (i_measured[:-1] + i_measured[1:]) * dt_s)])
    soc_cc = true_initial_soc - (q_integrated / capacity_as) * 100.0
    return soc_cc


def evaluate_cycle(
    df: pd.DataFrame,
    cycle_name: str,
    rate_nominal_a: float,
    model: PureNumpyMLP,
    v_noise_mv: float = 5.0,
    i_offset_ma: float = 30.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate ML model and Coulomb Counting on a single cycle run with hardware noise."""
    rng = np.random.default_rng(seed)
    v_raw = df["V"].to_numpy()
    i_raw = df["I"].to_numpy()
    y_true = df["soc_true"].to_numpy()
    
    # 1. Apply hardware noise chain
    v_noisy, i_noisy, vs_noisy = apply_hardware_noise_augmentation(
        v_raw, i_raw,
        v_noise_sigma_mv=v_noise_mv,
        i_offset_ma=i_offset_ma,
        quantize=True,
        rng=rng
    )
    
    # 2. Run ML inference (pure-NumPy)
    X_noisy = np.column_stack([v_noisy, i_noisy, vs_noisy])
    y_ml = model.predict_batch(X_noisy)
    
    # 3. Run Coulomb Counting baseline (with 30 mA biased sensor)
    # Estimate total capacity from the file duration and nominal current
    dt = 5.0
    # Capacity is total integrated true |I| dt
    true_capacity_as = np.sum(0.5 * (np.abs(i_raw[:-1]) + np.abs(i_raw[1:])) * dt)
    # If it is a pair, capacity is roughly the half-cycle (or use file's measured charge)
    # For a pair, CC tracks net charge flow relative to initial SOC
    # Capacity is the nominal charge capacity: true_capacity_as / 2 for pair, true_capacity_as for single
    is_pair = "pair" in cycle_name.lower()
    cycle_capacity = (true_capacity_as / 2.0) if is_pair else true_capacity_as
    
    y_cc = simulate_coulomb_counting(i_noisy, dt, y_true[0], cycle_capacity)
    
    # Error metrics
    err_ml = np.abs(y_true - y_ml)
    err_cc = np.abs(y_true - y_cc)
    
    ml_mae = float(mean_absolute_error(y_true, y_ml))
    ml_rmse = float(root_mean_squared_error(y_true, y_ml))
    ml_max = float(np.max(err_ml))
    
    cc_mae = float(mean_absolute_error(y_true, y_cc))
    cc_rmse = float(root_mean_squared_error(y_true, y_cc))
    cc_max = float(np.max(err_cc))
    
    # Discharging vs Charging split
    dis_mask = i_raw > 0.05
    chg_mask = i_raw < -0.05
    
    ml_mae_dis = float(mean_absolute_error(y_true[dis_mask], y_ml[dis_mask])) if np.any(dis_mask) else 0.0
    ml_mae_chg = float(mean_absolute_error(y_true[chg_mask], y_ml[chg_mask])) if np.any(chg_mask) else 0.0
    
    return {
        "cycle_name": cycle_name,
        "rate_a": rate_nominal_a,
        "samples": len(df),
        "duration_h": float(df["time_s"].iloc[-1] / 3600.0),
        "ml_mae": ml_mae,
        "ml_rmse": ml_rmse,
        "ml_max": ml_max,
        "cc_mae": cc_mae,
        "cc_rmse": cc_rmse,
        "cc_max": cc_max,
        "ml_mae_dis": ml_mae_dis,
        "ml_mae_chg": ml_mae_chg,
        "raw_series": {
            "time_h": (df["time_s"] / 3600.0).tolist(),
            "v": v_noisy.tolist(),
            "i": i_noisy.tolist(),
            "y_true": y_true.tolist(),
            "y_ml": y_ml.tolist(),
            "y_cc": y_cc.tolist(),
        }
    }


def run_full_evaluation() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Execute evaluation on all held-out test cycles."""
    config = load_config()
    dataset = load_cached_dataset()
    model = PureNumpyMLP()
    
    # Prepare held-out evaluation datasets
    test_runs = []
    
    # 1. Discharge 02 (0.5A discharge only)
    df_dis02 = dataset["Discharge_02.csv"]
    test_runs.append((df_dis02, "Discharge 02 (0.5 A)", 0.5))
    
    # 2. Pair 10 (1.0 A discharge + charge)
    pair10 = build_cycle_pair(dataset["Discharge_10.csv"], dataset["Load_10.csv"])
    test_runs.append((pair10, "Pair 10 (1.0 A Pair)", 1.0))
    
    # 3. Pair 20 (2.0 A discharge + charge)
    pair20 = build_cycle_pair(dataset["Discharge_20.csv"], dataset["Load_20.csv"])
    test_runs.append((pair20, "Pair 20 (2.0 A Pair)", 2.0))
    
    # 4. Pair 30 (3.0 A discharge + charge)
    pair30 = build_cycle_pair(dataset["Discharge_30.csv"], dataset["Load_30.csv"])
    test_runs.append((pair30, "Pair 30 (3.0 A Pair)", 3.0))
    
    results = []
    for df, name, rate in test_runs:
        res = evaluate_cycle(df, name, rate, model, v_noise_mv=5.0, i_offset_ma=30.0)
        results.append(res)
        
    summary_rows = []
    for r in results:
        summary_rows.append({
            "Cycle": r["cycle_name"],
            "Rate (A)": r["rate_a"],
            "ML MAE (%)": round(r["ml_mae"], 2),
            "ML RMSE (%)": round(r["ml_rmse"], 2),
            "ML Max (%)": round(r["ml_max"], 2),
            "CC MAE (%)": round(r["cc_mae"], 2),
            "CC RMSE (%)": round(r["cc_rmse"], 2),
            "CC Max (%)": round(r["cc_max"], 2),
            "ML MAE (Dis)": round(r["ml_mae_dis"], 2),
            "ML MAE (Chg)": round(r["ml_mae_chg"], 2),
        })
        
    res_df = pd.DataFrame(summary_rows)
    
    # Save CSV and Markdown tables
    tables_dir = get_path("tables_dir")
    res_df.to_csv(tables_dir / "held_out_evaluation.csv", index=False)
    
    with open(tables_dir / "held_out_evaluation.md", "w", encoding="utf-8") as f:
        f.write("# Held-Out Test Evaluation Results (With 5 mV ADC Noise & 30 mA Current Offset)\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n*Target: Mean Absolute Error < 2-3 percentage points on every cycle.*\n")
        
    # Generate high-resolution held-out comparison plots
    figures_dir = get_path("figures_dir")
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), dpi=300)
    
    for idx, r in enumerate(results):
        t = np.array(r["raw_series"]["time_h"])
        y_t = np.array(r["raw_series"]["y_true"])
        y_m = np.array(r["raw_series"]["y_ml"])
        y_c = np.array(r["raw_series"]["y_cc"])
        
        # Column 1: SOC vs Time
        ax_soc = axes[idx, 0]
        ax_soc.plot(t, y_t, color="#1f2937", lw=2.2, label="True SOC (Reference)")
        ax_soc.plot(t, y_m, color="#16a34a", lw=2.0, linestyle="--", label=f"ML 3-16-1 (MAE {r['ml_mae']:.2f}%)")
        ax_soc.plot(t, y_c, color="#dc2626", lw=1.5, linestyle=":", label=f"Coulomb Counting (MAE {r['cc_mae']:.2f}%)")
        ax_soc.set_title(f"{r['cycle_name']} - SOC Tracking", fontsize=11, fontweight="bold", color="#8c1236")
        ax_soc.set_xlabel("Time (hours)")
        ax_soc.set_ylabel("SOC (%)")
        ax_soc.set_ylim(-5, 105)
        ax_soc.grid(True, linestyle="--", alpha=0.5)
        ax_soc.legend(loc="upper right", fontsize=9)
        
        # Column 2: Error vs Time
        ax_err = axes[idx, 1]
        err_m = y_m - y_t
        err_c = y_c - y_t
        ax_err.plot(t, err_m, color="#16a34a", lw=1.8, label="ML Error")
        ax_err.plot(t, err_c, color="#dc2626", lw=1.2, linestyle=":", label="Coulomb Counting Error")
        ax_err.axhspan(-3.0, 3.0, color="#16a34a", alpha=0.15, label="±3% Target Band")
        ax_err.set_title(f"{r['cycle_name']} - Estimation Error (Percentage Points)", fontsize=11, fontweight="bold", color="#8c1236")
        ax_err.set_xlabel("Time (hours)")
        ax_err.set_ylabel("Error (pts)")
        ax_err.set_ylim(-10, 10)
        ax_err.grid(True, linestyle="--", alpha=0.5)
        ax_err.legend(loc="upper right", fontsize=9)
        
    plt.tight_layout()
    fig.savefig(figures_dir / "held_out_tracking.png", dpi=300)
    plt.close(fig)
    print(f"[Eval] Saved held-out comparison figure to {figures_dir / 'held_out_tracking.png'}")
    
    return res_df, results


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3: HELD-OUT EVALUATION (WITH HARDWARE NOISE & SENSOR OFFSET)")
    print("=" * 70)
    df, _ = run_full_evaluation()
    print("\nEvaluation Results Table:")
    print(df.to_string(index=False))
