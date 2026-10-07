"""Experiments G & H: Error Analysis and Model Explainability.

Produces:
1. Error breakdown across SOC bins ([0-20], [20-40], [40-60], [60-80], [80-100]%)
2. Error in CC vs CV charge phases
3. Permutation feature importance analysis
4. Partial dependence curves: SOC vs Voltage at fixed currents (-2A, -1A, 0A, 1A, 2A)
5. Weight matrices visual (W1, W2 heatmaps)
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


def run_explain_experiment():
    print("=" * 70)
    print("EXPERIMENTS G & H: ERROR ANALYSIS & EXPLAINABILITY")
    print("=" * 70)
    
    dataset = load_cached_dataset()
    model = PureNumpyMLP()
    
    # 1. Error Analysis on Held-Out Test Set (Pair 10 + Pair 20 + Pair 30)
    all_pairs = [
        build_cycle_pair(dataset["Discharge_10.csv"], dataset["Load_10.csv"]),
        build_cycle_pair(dataset["Discharge_20.csv"], dataset["Load_20.csv"]),
        build_cycle_pair(dataset["Discharge_30.csv"], dataset["Load_30.csv"]),
    ]
    all_df = pd.concat(all_pairs, ignore_index=True)
    
    v_n, i_n, vs_n = apply_hardware_noise_augmentation(
        all_df["V"].to_numpy(), all_df["I"].to_numpy(),
        v_noise_sigma_mv=5.0, i_offset_ma=30.0, rng=np.random.default_rng(42)
    )
    y_true = all_df["soc_true"].to_numpy()
    preds = model.predict_batch(np.column_stack([v_n, i_n, vs_n]))
    abs_errors = np.abs(y_true - preds)
    
    # SOC Bins
    bins = [0, 20, 40, 60, 80, 100]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    bin_maes = []
    for k in range(len(bins) - 1):
        mask = (y_true >= bins[k]) & (y_true <= bins[k + 1])
        bin_maes.append(float(np.mean(abs_errors[mask])) if np.any(mask) else 0.0)
        
    error_bin_df = pd.DataFrame({"SOC_Bin": bin_labels, "MAE_pct": np.round(bin_maes, 2)})
    
    # CC vs CV phase in charging
    chg_mask = all_df["I"] < -0.05
    cc_mask = chg_mask & (all_df["V"] < 4.18)
    cv_mask = chg_mask & (all_df["V"] >= 4.18)
    mae_cc_phase = float(np.mean(abs_errors[cc_mask]))
    mae_cv_phase = float(np.mean(abs_errors[cv_mask]))
    
    print("\n[Error Analysis]")
    print(error_bin_df.to_string(index=False))
    print(f"Charging CC Phase MAE: {mae_cc_phase:.2f}% | CV Tapering Phase MAE: {mae_cv_phase:.2f}%")
    
    # 2. Permutation Feature Importance
    baseline_mae = mean_absolute_error(y_true, preds)
    X_base = np.column_stack([v_n, i_n, vs_n])
    rng = np.random.default_rng(123)
    
    feat_names = ["V (Terminal Voltage)", "I (Current)", "V_smooth (60s Causal MA)"]
    importance_scores = []
    for col_idx in range(3):
        X_perm = X_base.copy()
        X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
        perm_preds = model.predict_batch(X_perm)
        perm_mae = mean_absolute_error(y_true, perm_preds)
        delta_mae = perm_mae - baseline_mae
        importance_scores.append(float(delta_mae))
        print(f"Feature '{feat_names[col_idx]}' Importance (Delta MAE): +{delta_mae:.2f}%")
        
    pfi_df = pd.DataFrame({"Feature": feat_names, "Importance_Delta_MAE_pct": np.round(importance_scores, 2)})
    
    # 3. Partial Dependence: SOC vs Voltage at fixed currents
    v_grid = np.linspace(3.0, 4.2, 100)
    currents_sweep = [-2.0, -1.0, 0.0, 1.0, 2.0]
    pd_curves = {}
    for i_fixed in currents_sweep:
        # Evaluate with V_smooth = V
        X_eval = np.column_stack([v_grid, np.full_like(v_grid, i_fixed), v_grid])
        pd_curves[i_fixed] = model.predict_batch(X_eval)
        
    # Save CSV tables
    tables_dir = get_path("tables_dir")
    error_bin_df.to_csv(tables_dir / "error_by_soc_bins.csv", index=False)
    pfi_df.to_csv(tables_dir / "permutation_feature_importance.csv", index=False)
    
    # Generate 4-panel Explainability figure
    figures_dir = get_path("figures_dir")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    
    # Panel 1: Error vs SOC Bins
    axes[0, 0].bar(error_bin_df["SOC_Bin"], error_bin_df["MAE_pct"], color="#8c1236", edgecolor="black", alpha=0.8)
    axes[0, 0].axhline(y=2.0, color="#16a34a", linestyle="--", label="2% Target")
    axes[0, 0].set_title("Estimation Error Across SOC Bins", fontsize=11, fontweight="bold", color="#8c1236")
    axes[0, 0].set_ylabel("MAE (%)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    
    # Panel 2: Permutation Feature Importance
    axes[0, 1].barh(pfi_df["Feature"], pfi_df["Importance_Delta_MAE_pct"], color="#2563eb", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title("Permutation Feature Importance (MAE Drop)", fontsize=11, fontweight="bold", color="#8c1236")
    axes[0, 1].set_xlabel("Increase in MAE when Shuffled (% points)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 3: Partial Dependence Curves
    for i_val, curve in pd_curves.items():
        axes[1, 0].plot(v_grid, curve, label=f"I = {i_val:+.1f} A", lw=2.0)
    axes[1, 0].set_title("Partial Dependence: Estimated SOC vs Terminal Voltage", fontsize=11, fontweight="bold", color="#8c1236")
    axes[1, 0].set_xlabel("Cell Voltage (V)")
    axes[1, 0].set_ylabel("Estimated SOC (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()
    
    # Panel 4: Hidden Layer Weight Matrix Heatmap
    im = axes[1, 1].imshow(model.W1, cmap="coolwarm", aspect="auto")
    axes[1, 1].set_title("Hidden Layer Weights W1 (16 Neurons × 3 Inputs)", fontsize=11, fontweight="bold", color="#8c1236")
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_xticklabels(["V", "I", "V_smooth"])
    axes[1, 1].set_ylabel("Hidden Neuron (1-16)")
    plt.colorbar(im, ax=axes[1, 1], label="Weight Value")
    
    plt.tight_layout()
    fig.savefig(figures_dir / "explainability_analysis.png", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved {figures_dir / 'explainability_analysis.png'}")
    print("[Explain] Completed successfully!")


if __name__ == "__main__":
    run_explain_experiment()
