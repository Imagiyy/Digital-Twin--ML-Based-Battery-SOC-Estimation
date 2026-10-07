"""Experiment E: Robustness Sweeps Under Hardware Stress.

Sweeps:
1. ADC Noise: 0 to 50 mV (step 5 mV)
2. Current Sensor Offset: 0 to 200 mA (step 20 mA)
3. ADC Resolution: 8, 10, 12, 14, 16 bits
4. Resistor Divider Gain Error: -2% to +2%

Evaluates the canonical trained MLP 3-16-1 against the 2-3% accuracy target line.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from src.config import get_path
from src.data_prep import load_cached_dataset, build_cycle_pair
from src.mlp_infer import PureNumpyMLP
from src.train import apply_hardware_noise_augmentation


def run_robustness_experiment():
    print("=" * 70)
    print("EXPERIMENT E: HARDWARE ROBUSTNESS SWEEPS")
    print("=" * 70)
    
    dataset = load_cached_dataset()
    model = PureNumpyMLP()
    
    # Test on Pair 10
    pair10 = build_cycle_pair(dataset["Discharge_10.csv"], dataset["Load_10.csv"])
    v_raw = pair10["V"].to_numpy()
    i_raw = pair10["I"].to_numpy()
    y_true = pair10["soc_true"].to_numpy()
    
    # 1. Sweep ADC Noise (0 to 50 mV)
    noise_vals = np.arange(0.0, 55.0, 5.0)
    noise_maes = []
    for nv in noise_vals:
        v_n, i_n, vs_n = apply_hardware_noise_augmentation(
            v_raw, i_raw, v_noise_sigma_mv=nv, i_offset_ma=30.0, rng=np.random.default_rng(42)
        )
        preds = model.predict_batch(np.column_stack([v_n, i_n, vs_n]))
        noise_maes.append(mean_absolute_error(y_true, preds))
        
    # 2. Sweep Current Offset (0 to 200 mA)
    offset_vals = np.arange(0.0, 220.0, 20.0)
    offset_maes = []
    for ov in offset_vals:
        v_n, i_n, vs_n = apply_hardware_noise_augmentation(
            v_raw, i_raw, v_noise_sigma_mv=5.0, i_offset_ma=ov, rng=np.random.default_rng(42)
        )
        preds = model.predict_batch(np.column_stack([v_n, i_n, vs_n]))
        offset_maes.append(mean_absolute_error(y_true, preds))
        
    # 3. Sweep ADC Bits (8, 10, 12, 14, 16)
    bits_vals = [8, 10, 12, 14, 16]
    bits_maes = []
    for b in bits_vals:
        v_n, i_n, vs_n = apply_hardware_noise_augmentation(
            v_raw, i_raw, v_noise_sigma_mv=5.0, i_offset_ma=30.0, bits=b, rng=np.random.default_rng(42)
        )
        preds = model.predict_batch(np.column_stack([v_n, i_n, vs_n]))
        bits_maes.append(mean_absolute_error(y_true, preds))
        
    # 4. Sweep Divider Gain Error (-2% to +2%)
    gain_err_vals = np.linspace(-0.02, 0.02, 9)
    gain_maes = []
    for ge in gain_err_vals:
        v_n, i_n, vs_n = apply_hardware_noise_augmentation(
            v_raw * (1.0 + ge), i_raw, v_noise_sigma_mv=5.0, i_offset_ma=30.0, rng=np.random.default_rng(42)
        )
        preds = model.predict_batch(np.column_stack([v_n, i_n, vs_n]))
        gain_maes.append(mean_absolute_error(y_true, preds))
        
    # Save CSV tables
    tables_dir = get_path("tables_dir")
    pd.DataFrame({"ADC_Noise_mV": noise_vals, "MAE_pct": noise_maes}).to_csv(tables_dir / "robustness_noise.csv", index=False)
    pd.DataFrame({"Offset_mA": offset_vals, "MAE_pct": offset_maes}).to_csv(tables_dir / "robustness_offset.csv", index=False)
    pd.DataFrame({"ADC_Bits": bits_vals, "MAE_pct": bits_maes}).to_csv(tables_dir / "robustness_bits.csv", index=False)
    pd.DataFrame({"Gain_Error_pct": gain_err_vals * 100.0, "MAE_pct": gain_maes}).to_csv(tables_dir / "robustness_gain.csv", index=False)
    
    # Generate 4-panel figure
    figures_dir = get_path("figures_dir")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    
    # Subplot 1: Noise
    axes[0, 0].plot(noise_vals, noise_maes, "o-", color="#8c1236", lw=2.0)
    axes[0, 0].axhline(y=2.0, color="#16a34a", linestyle="--", label="2% Target")
    axes[0, 0].axhline(y=3.0, color="#dc2626", linestyle="--", label="3% Bound")
    axes[0, 0].set_title("ADC Noise Sweep (0 - 50 mV)", fontsize=11, fontweight="bold", color="#8c1236")
    axes[0, 0].set_xlabel("Noise Sigma on ADC Pin (mV)")
    axes[0, 0].set_ylabel("MAE (%)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    
    # Subplot 2: Offset
    axes[0, 1].plot(offset_vals, offset_maes, "s-", color="#2563eb", lw=2.0)
    axes[0, 1].axhline(y=2.0, color="#16a34a", linestyle="--", label="2% Target")
    axes[0, 1].axhline(y=3.0, color="#dc2626", linestyle="--", label="3% Bound")
    axes[0, 1].set_title("Current Sensor Offset Sweep (0 - 200 mA)", fontsize=11, fontweight="bold", color="#8c1236")
    axes[0, 1].set_xlabel("Sensor DC Bias Offset (mA)")
    axes[0, 1].set_ylabel("MAE (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()
    
    # Subplot 3: Bits
    axes[1, 0].plot(bits_vals, bits_maes, "^-", color="#16a34a", lw=2.0)
    axes[1, 0].axhline(y=2.0, color="#16a34a", linestyle="--", label="2% Target")
    axes[1, 0].axhline(y=3.0, color="#dc2626", linestyle="--", label="3% Bound")
    axes[1, 0].set_title("ADC Resolution Sweep (8 - 16 Bits)", fontsize=11, fontweight="bold", color="#8c1236")
    axes[1, 0].set_xlabel("ADC Bits")
    axes[1, 0].set_ylabel("MAE (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()
    
    # Subplot 4: Divider Gain Error
    axes[1, 1].plot(gain_err_vals * 100.0, gain_maes, "d-", color="#d97706", lw=2.0)
    axes[1, 1].axhline(y=2.0, color="#16a34a", linestyle="--", label="2% Target")
    axes[1, 1].axhline(y=3.0, color="#dc2626", linestyle="--", label="3% Bound")
    axes[1, 1].set_title("Resistor Divider Gain Error (±2%)", fontsize=11, fontweight="bold", color="#8c1236")
    axes[1, 1].set_xlabel("Divider Gain Tolerance (%)")
    axes[1, 1].set_ylabel("MAE (%)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()
    
    plt.tight_layout()
    fig.savefig(figures_dir / "robustness_sweeps.png", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved {figures_dir / 'robustness_sweeps.png'}")
    print("[Robustness] Completed successfully!")


if __name__ == "__main__":
    run_robustness_experiment()
