"""Model Training Pipeline with Noise Augmentation and Firmware Export (Phase 3).

Trains the canonical MLP 3-16-1 State of Charge estimator.
Includes:
- Strict file-level train/validation isolation (held-out test set untouched)
- Noise augmentation (Gaussian ADC noise 0-10 mV, sequence current offset 0-50 mA, ADC quantisation)
- Dynamic re-computation of V_smooth on noisy voltage
- Learning rate schedule and early stopping on validation MAE
- Generation of training curves figure (300 DPI) and CSV log
- Export of weights to models/mlp_weights.json and models/mlp_weights.h
"""

from pathlib import Path
import json
import joblib
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import get_path, load_config
from src.data_prep import load_cached_dataset, compute_causal_moving_average
from src.split import get_dataset_splits, fit_scaler_on_train
from src.model import MLP3_16_1


def apply_hardware_noise_augmentation(
    v_clean: np.ndarray,
    i_clean: np.ndarray,
    v_noise_sigma_mv: float = 5.0,
    i_offset_ma: float = 30.0,
    quantize: bool = True,
    vref: float = 3.3,
    ratio: float = 0.5,
    bits: int = 12,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate hardware chain noise on battery signals during training.
    
    1. Vpin = V * ratio + Gaussian_noise(sigma_mv / 1000)
    2. Quantise to ADC codes: round(Vpin / Vref * (2^bits - 1))
    3. Reconstruct V_adc = code / (2^bits - 1) * Vref / ratio
    4. Current: I_sensor = I + offset_ma / 1000
    5. Recompute V_smooth from V_adc using identical causal expanding window
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Divider + ADC noise on pin
    v_pin_clean = v_clean * ratio
    noise_v = rng.normal(0.0, (v_noise_sigma_mv / 1000.0), size=len(v_clean))
    v_pin_noisy = v_pin_clean + noise_v
    
    if quantize:
        max_code = (1 << bits) - 1
        codes = np.clip(np.round((v_pin_noisy / vref) * max_code), 0, max_code)
        v_recon = (codes / max_code) * (vref / ratio)
    else:
        v_recon = v_pin_noisy / ratio
        
    # Current sensor offset
    i_noisy = i_clean + (i_offset_ma / 1000.0)
    
    # Recompute causal 60s moving average on the reconstructed noisy voltage
    v_smooth_noisy = compute_causal_moving_average(v_recon, window_samples=12)
    
    return v_recon, i_noisy, v_smooth_noisy


def build_augmented_training_set(
    train_dfs: List[pd.DataFrame],
    copies: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented dataset with random noise sigma and current offsets per sequence."""
    rng = np.random.default_rng(seed)
    all_features = []
    all_soc = []
    
    for df in train_dfs:
        v_raw = df["V"].to_numpy(dtype=np.float64)
        i_raw = df["I"].to_numpy(dtype=np.float64)
        soc_raw = df["soc_true"].to_numpy(dtype=np.float64)
        
        # 1. Clean copy
        all_features.append(df[["V", "I", "V_smooth"]].to_numpy())
        all_soc.append(soc_raw)
        
        # 2. Augmented copies with randomized physical parameters
        for _ in range(copies - 1):
            sigma_mv = rng.uniform(2.0, 10.0)
            offset_ma = rng.uniform(-50.0, 50.0)
            v_aug, i_aug, vs_aug = apply_hardware_noise_augmentation(
                v_raw, i_raw,
                v_noise_sigma_mv=sigma_mv,
                i_offset_ma=offset_ma,
                quantize=True,
                rng=rng,
            )
            feat_aug = np.column_stack([v_aug, i_aug, vs_aug])
            all_features.append(feat_aug)
            all_soc.append(soc_raw)
            
    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_soc)
    return X_train, y_train


def train_model(
    max_iter: int = 80,
    seed: int = 42,
) -> Tuple[MLP3_16_1, Dict[str, Any]]:
    """Execute training pipeline and return trained model instance with metrics."""
    config = load_config()
    dataset = load_cached_dataset()
    splits = get_dataset_splits(list(dataset.keys()))
    
    print(f"[Train] Splitting: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} held-out test.")
    
    # Extract train and val dataframes
    train_dfs = [dataset[name] for name in splits["train"]]
    val_dfs = [dataset[name] for name in splits["val"]]
    
    # 1. Fit scaler strictly on training files
    scaler_dict = fit_scaler_on_train(train_dfs)
    scaler_mean = np.array(scaler_dict["mean"], dtype=np.float32)
    scaler_std = np.array(scaler_dict["std"], dtype=np.float32)
    
    # 2. Build augmented training dataset
    print("[Train] Generating noise-augmented training samples...")
    X_train_raw, y_train = build_augmented_training_set(train_dfs, copies=4, seed=seed)
    print(f"[Train] Training set size: {len(X_train_raw):,} samples (target ~300k).")
    
    # 3. Build validation set (with standard hardware noise)
    val_features = []
    val_soc = []
    rng_val = np.random.default_rng(123)
    for df in val_dfs:
        v_val, i_val, vs_val = apply_hardware_noise_augmentation(
            df["V"].to_numpy(), df["I"].to_numpy(),
            v_noise_sigma_mv=5.0, i_offset_ma=30.0,
            rng=rng_val
        )
        val_features.append(np.column_stack([v_val, i_val, vs_val]))
        val_soc.append(df["soc_true"].to_numpy())
    X_val_raw = np.vstack(val_features)
    y_val = np.concatenate(val_soc)
    
    # 4. Normalize inputs using training scaler
    X_train_norm = (X_train_raw - scaler_mean) / scaler_std
    X_val_norm = (X_val_raw - scaler_mean) / scaler_std
    
    # 5. Initialize and train MLPRegressor (3 -> 16 -> 1, tanh)
    print("[Train] Fitting MLP 3-16-1 with tanh hidden units and Adam optimizer...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="tanh",
        solver="adam",
        alpha=0.0001, # L2 regularisation
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=1, # we iterate epoch-by-epoch for custom tracking & curves
        warm_start=True,
        random_state=seed,
        shuffle=True,
    )
    
    train_losses = []
    val_maes = []
    val_rmses = []
    best_val_mae = float("inf")
    best_weights = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, max_iter + 1):
        mlp.fit(X_train_norm, y_train)
        
        # Predictions
        y_val_pred = np.clip(mlp.predict(X_val_norm), 0.0, 100.0)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        
        train_losses.append(float(mlp.loss_))
        val_maes.append(float(val_mae))
        val_rmses.append(float(val_rmse))
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = (
                mlp.coefs_[0].copy(), # (3, 16)
                mlp.intercepts_[0].copy(), # (16,)
                mlp.coefs_[1].copy().flatten(), # (16,)
                float(mlp.intercepts_[1][0]), # scalar
            )
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{max_iter:02d} | Train Loss (MSE): {mlp.loss_:.4f} | Val MAE: {val_mae:.2f}% | Val RMSE: {val_rmse:.2f}%")
            
        if patience_counter >= patience:
            print(f"[Train] Early stopping triggered at epoch {epoch} (best val MAE: {best_val_mae:.2f}%).")
            break
            
    # Unpack best weights
    W1_raw, b1_raw, W2_raw, b2_raw = best_weights
    
    # Restore best weights to sklearn model so joblib artifact matches exported JSON exactly
    mlp.coefs_[0] = W1_raw
    mlp.intercepts_[0] = b1_raw
    mlp.coefs_[1] = W2_raw.reshape(-1, 1)
    mlp.intercepts_[1] = np.array([b2_raw], dtype=np.float64)
    
    # Format model instance
    trained_model = MLP3_16_1(
        W1=W1_raw.T, # shape (16, 3)
        b1=b1_raw,
        W2=W2_raw,
        b2=b2_raw,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    
    # 6. Save training curves
    history_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_mae": val_maes,
        "val_rmse": val_rmses,
    })
    tables_dir = get_path("tables_dir")
    history_df.to_csv(tables_dir / "training_curves.csv", index=False)
    
    figures_dir = get_path("figures_dir")
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    color = "#8c1236"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Train Loss (MSE)", color=color, fontsize=12)
    ax1.plot(history_df["epoch"], history_df["train_loss"], color=color, lw=2.0, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color = "#16a34a"
    ax2.set_ylabel("Validation MAE (%)", color=color, fontsize=12)
    ax2.plot(history_df["epoch"], history_df["val_mae"], color=color, lw=2.0, linestyle="--", label="Val MAE")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("MLP 3-16-1 Training Convergence & Validation Error", fontsize=13, fontweight="bold", color="#8c1236")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved training curves to {figures_dir / 'training_curves.png'}")
    
    # 7. Model export
    models_dir = get_path("models_dir")
    firmware_include_dir = get_path("firmware_dir") / "include"
    
    metadata = {
        "train_samples": len(X_train_raw),
        "best_val_mae": float(best_val_mae),
        "epochs_trained": len(train_losses),
        "train_rmse": float(np.sqrt(train_losses[-1])),
    }
    trained_model.export_weights_json(models_dir / "mlp_weights.json", metadata)
    trained_model.export_weights_c_header(models_dir / "mlp_weights.h")
    trained_model.export_weights_c_header(firmware_include_dir / "mlp_weights.h")
    
    # Save scikit-learn model artifact
    joblib.dump(mlp, models_dir / "mlp_model.joblib")
    print(f"[Model] Saved joblib model to {models_dir / 'mlp_model.joblib'}")
    print(f"[Model] Parameter verification: {trained_model.total_parameters} trainable parameters.")
    
    return trained_model, metadata


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3: MODEL TRAINING & ARTIFACT EXPORT")
    print("=" * 70)
    model, meta = train_model()
    print("Training finished successfully!")
