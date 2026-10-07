"""Experiment A: Baselines Comparison.

Compares:
1. Proposed MLP 3-16-1
2. Coulomb Counting (Biased 30 mA sensor, True initial SOC)
3. Coulomb Counting (Biased 30 mA sensor, Wrong initial SOC -10%)
4. OCV / Voltage-only Interpolation Table Lookup
5. Linear Regression on [V, I, V_smooth]
6. Random Forest Regressor (Offline reference)

Evaluated under standard hardware noise (5 mV ADC noise, 30 mA current offset).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import get_path
from src.data_prep import load_cached_dataset, build_cycle_pair
from src.split import get_dataset_splits, fit_scaler_on_train
from src.mlp_infer import PureNumpyMLP
from src.train import apply_hardware_noise_augmentation
from src.evaluate import simulate_coulomb_counting
from src.kalman import BatteryECM, ExtendedKalmanFilter, SigmaPointKalmanFilter


def run_baselines_experiment():
    print("=" * 70)
    print("EXPERIMENT A: BASELINES COMPARISON")
    print("=" * 70)
    
    dataset = load_cached_dataset()
    splits = get_dataset_splits(list(dataset.keys()))
    
    # 1. Train linear regression and random forest baselines on training set
    train_dfs = [dataset[name] for name in splits["train"]]
    scaler = fit_scaler_on_train(train_dfs)
    scaler_mean = np.array(scaler["mean"])
    scaler_std = np.array(scaler["std"])
    
    X_train_list = []
    y_train_list = []
    for df in train_dfs:
        X_train_list.append(df[["V", "I", "V_smooth"]].to_numpy())
        y_train_list.append(df["soc_true"].to_numpy())
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    # Fit Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit((X_train - scaler_mean) / scaler_std, y_train)
    
    # Fit Random Forest (subsampled for speed)
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train[::5], y_train[::5])
    
    # 2. Build OCV lookup curve from low-rate 0.5A discharge (Discharge_01)
    df_ocv = dataset["Discharge_01.csv"]
    v_ocv_ref = df_ocv["V"].to_numpy()
    soc_ocv_ref = df_ocv["soc_true"].to_numpy()
    # Sort for monotonic interpolation
    sort_idx = np.argsort(v_ocv_ref)
    v_sorted = v_ocv_ref[sort_idx]
    soc_sorted = soc_ocv_ref[sort_idx]
    
    # 3. Load trained MLP
    mlp = PureNumpyMLP()
    
    # 4. Prepare held-out test cycles
    test_cycles = [
        ("Discharge 02 (0.5 A)", dataset["Discharge_02.csv"], 0.5, False),
        ("Pair 10 (1.0 A)", build_cycle_pair(dataset["Discharge_10.csv"], dataset["Load_10.csv"]), 1.0, True),
        ("Pair 20 (2.0 A)", build_cycle_pair(dataset["Discharge_20.csv"], dataset["Load_20.csv"]), 2.0, True),
        ("Pair 30 (3.0 A)", build_cycle_pair(dataset["Discharge_30.csv"], dataset["Load_30.csv"]), 3.0, True),
    ]
    
    records = []
    for cycle_title, df, rate, is_pair in test_cycles:
        # Hardware noise
        v_noisy, i_noisy, vs_noisy = apply_hardware_noise_augmentation(
            df["V"].to_numpy(), df["I"].to_numpy(),
            v_noise_sigma_mv=5.0, i_offset_ma=30.0,
            rng=np.random.default_rng(42)
        )
        X_test = np.column_stack([v_noisy, i_noisy, vs_noisy])
        y_true = df["soc_true"].to_numpy()
        dt = 5.0
        
        # 1. Proposed MLP
        pred_mlp = mlp.predict_batch(X_test)
        
        # 2. Coulomb Counting (True Initial SOC)
        tot_q = np.sum(0.5 * (np.abs(df["I"].iloc[:-1].to_numpy()) + np.abs(df["I"].iloc[1:].to_numpy())) * dt)
        cap = (tot_q / 2.0) if is_pair else tot_q
        pred_cc = simulate_coulomb_counting(i_noisy, dt, y_true[0], cap)
        
        # 3. Coulomb Counting (Wrong Initial SOC -10%)
        pred_cc_wrong = simulate_coulomb_counting(i_noisy, dt, max(0.0, y_true[0] - 10.0), cap)
        
        # 4. EKF (Extended Kalman Filter)
        ecm = BatteryECM(capacity_ah=cap / 3600.0, dt_s=dt)
        ekf = ExtendedKalmanFilter(initial_soc=y_true[0] / 100.0, ecm=ecm)
        pred_ekf = np.array([ekf.step(v, i)["soc_ekf"] for v, i in zip(v_noisy, i_noisy)])

        # 5. SPKF (Sigma-Point Kalman Filter / UKF)
        spkf = SigmaPointKalmanFilter(initial_soc=y_true[0] / 100.0, ecm=ecm)
        pred_spkf = np.array([spkf.step(v, i)["soc_spkf"] for v, i in zip(v_noisy, i_noisy)])

        # 6. OCV Lookup (V only)
        pred_ocv = np.interp(v_noisy, v_sorted, soc_sorted, left=0.0, right=100.0)
        
        # 7. Linear Regression
        pred_lin = np.clip(lin_reg.predict((X_test - scaler_mean) / scaler_std), 0.0, 100.0)
        
        # 8. Random Forest
        pred_rf = np.clip(rf.predict(X_test), 0.0, 100.0)
        
        models = [
            ("MLP 3-16-1 (Proposed)", pred_mlp),
            ("SPKF (Sigma-Point Kalman)", pred_spkf),
            ("EKF (Extended Kalman)", pred_ekf),
            ("Coulomb Counting (True SOC0)", pred_cc),
            ("Coulomb Counting (Wrong SOC0 -10%)", pred_cc_wrong),
            ("OCV Table Lookup (V-only)", pred_ocv),
            ("Linear Regression", pred_lin),
        ]
        
        for name, pred in models:
            mae = mean_absolute_error(y_true, pred)
            rmse = root_mean_squared_error(y_true, pred)
            max_err = np.max(np.abs(y_true - pred))
            records.append({
                "Cycle": cycle_title,
                "Model": name,
                "MAE (%)": round(float(mae), 2),
                "RMSE (%)": round(float(rmse), 2),
                "Max Error (%)": round(float(max_err), 2),
            })
            
    res_df = pd.DataFrame(records)
    
    # Save table
    tables_dir = get_path("tables_dir")
    res_df.to_csv(tables_dir / "baselines_comparison.csv", index=False)
    with open(tables_dir / "baselines_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Baselines Comparison Table\n\n")
        f.write(res_df.to_markdown(index=False))
        
    # Generate bar chart
    figures_dir = get_path("figures_dir")
    pivot_mae = res_df.pivot(index="Cycle", columns="Model", values="MAE (%)")
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    pivot_mae.plot(kind="bar", ax=ax, colormap="tab10", width=0.8, edgecolor="black", alpha=0.85)
    ax.axhline(y=3.0, color="#dc2626", linestyle="--", lw=1.5, label="3% Accuracy Target")
    ax.set_title("Baselines Comparison Across Held-Out Test Cycles (MAE %)", fontsize=13, fontweight="bold", color="#8c1236")
    ax.set_ylabel("Mean Absolute Error (%)", fontsize=12)
    ax.set_xlabel("")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(figures_dir / "baselines_comparison.png", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved {figures_dir / 'baselines_comparison.png'}")
    print("[Baselines] Completed successfully!")


if __name__ == "__main__":
    run_baselines_experiment()
