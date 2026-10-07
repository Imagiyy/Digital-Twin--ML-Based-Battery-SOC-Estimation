"""Digital Twin Orchestrator and Headless Simulation Runner (Phase 4).

Connects the full virtual hardware signal flow:
NASA cell data -> VoltageDivider -> ESP32ADC -> CurrentSensor -> Esp32Firmware -> TP4056
Runs alongside reference estimators:
- True SOC label
- Coulomb Counting with biased sensor (30 mA)
- Coulomb Counting with biased sensor + WRONG initial SOC (±10%)

Provides:
- Step-by-step execution for WebSocket backend
- Headless CLI runner: python -m src.twin --cycle pair10 --noise 5 --offset 30
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data_prep import load_cached_dataset, build_cycle_pair
from src.mlp_infer import PureNumpyMLP
from src.hardware.cell import CellTwin
from src.hardware.divider import VoltageDivider
from src.hardware.adc import ESP32ADC
from src.hardware.current_sensor import CurrentSensor
from src.hardware.tp4056 import TP4056, TP4056State
from src.hardware.esp32 import Esp32Firmware
from src.kalman import BatteryECM, ExtendedKalmanFilter, SigmaPointKalmanFilter
from src.soh import CombinedSOHEstimator


class DigitalTwin:
    """Complete Digital Twin of the ESP32 + TP4056 + 18650 Battery Monitoring System."""
    
    def __init__(
        self,
        cycle_df: pd.DataFrame,
        cycle_name: str = "pair10",
        model: Optional[PureNumpyMLP] = None,
        adc_noise_mv: float = 5.0,
        current_offset_ma: float = 30.0,
        adc_bits: int = 12,
        seed: int = 42,
    ):
        self.cycle_name = cycle_name
        self.rng = np.random.default_rng(seed)
        
        # Load model if not provided
        if model is None:
            weights_path = get_path("models_dir") / "mlp_weights.json"
            model = PureNumpyMLP(weights_path)
        self.model = model
        
        # Instantiate virtual hardware chain
        self.cell = CellTwin(cycle_df, cycle_name)
        self.divider = VoltageDivider(ratio=0.5)
        self.adc = ESP32ADC(vref=3.3, bits=adc_bits, noise_sigma_mv=adc_noise_mv, divider_ratio=0.5, rng=self.rng)
        self.current_sensor = CurrentSensor(offset_ma=current_offset_ma, rng=self.rng)
        initial_i = float(cycle_df["I"].iloc[0]) if len(cycle_df) > 0 else 0.0
        self.tp4056 = TP4056()
        if initial_i < -0.05:
            self.tp4056.current_state = TP4056State.CHARGING
            self.tp4056._target_state = TP4056State.CHARGING
        self.firmware = Esp32Firmware(model, window_size=12)
        
        # Coulomb counters state
        if len(cycle_df) > 1 and "time_s" in cycle_df.columns:
            diffs = np.diff(cycle_df["time_s"].to_numpy())
            pos_diffs = diffs[diffs > 0]
            self.dt_s = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 5.0
        else:
            self.dt_s = 5.0
        # Determine nominal capacity
        v_arr = cycle_df["V"].to_numpy()
        i_arr = cycle_df["I"].to_numpy()
        tot_q = np.sum(0.5 * (np.abs(i_arr[:-1]) + np.abs(i_arr[1:])) * self.dt_s)
        self.capacity_as = (tot_q / 2.0) if "pair" in cycle_name.lower() else tot_q
        if self.capacity_as <= 0.0:
            self.capacity_as = 2.6 * 3600.0 # fallback nominal 2.6 Ah
            
        initial_soc = 100.0
        if len(cycle_df) > 0:
            if "soc_true" in cycle_df.columns and not pd.isna(cycle_df["soc_true"].iloc[0]):
                initial_soc = float(cycle_df["soc_true"].iloc[0])
            elif "soc_ml" in cycle_df.columns and not pd.isna(cycle_df["soc_ml"].iloc[0]):
                initial_soc = float(cycle_df["soc_ml"].iloc[0])
        self.cc_true_soc = initial_soc
        self.cc_biased_soc = initial_soc
        self.cc_wrong_soc = min(100.0, max(0.0, initial_soc - 10.0)) # 10% offset
        
        # Kalman filters (EKF and SPKF)
        self.ecm = BatteryECM(capacity_ah=self.capacity_as / 3600.0, dt_s=self.dt_s)
        self.ekf = ExtendedKalmanFilter(initial_soc=initial_soc / 100.0, ecm=self.ecm)
        self.spkf = SigmaPointKalmanFilter(initial_soc=initial_soc / 100.0, ecm=self.ecm)
        
        # Dual SOH Estimator (Capacity Fade & Resistance Fade)
        cycle_idx = 10
        for pattern, num in [("pair30", 30), ("pair20", 20), ("pair10", 10), ("dis02", 2), ("30", 30), ("20", 20), ("10", 10), ("2", 2)]:
            if pattern in cycle_name.lower():
                cycle_idx = num
                break
        self.soh = CombinedSOHEstimator(cycle_number=cycle_idx)
        
        # Metric tracking
        self.history_true_soc = []
        self.history_ml_soc = []
        self.history_ekf_soc = []
        self.history_spkf_soc = []
        self.history_cc_soc = []
        self.history_cc_wrong_soc = []
        self.history_soh = []
        
    def reset(self) -> None:
        """Reset all hardware components and estimators."""
        self.cell.reset()
        self.firmware.reset()
        initial_i = float(self.cell.df["I"].iloc[0]) if len(self.cell.df) > 0 and "I" in self.cell.df.columns else 0.0
        init_state = TP4056State.CHARGING if initial_i < -0.05 else TP4056State.DISCHARGING
        self.tp4056.reset(initial_state=init_state)

        initial_soc = 100.0
        if len(self.cell.df) > 0:
            if "soc_true" in self.cell.df.columns and not pd.isna(self.cell.df["soc_true"].iloc[0]):
                initial_soc = float(self.cell.df["soc_true"].iloc[0])
            elif "soc_ml" in self.cell.df.columns and not pd.isna(self.cell.df["soc_ml"].iloc[0]):
                initial_soc = float(self.cell.df["soc_ml"].iloc[0])

        self.cc_biased_soc = initial_soc
        self.cc_wrong_soc = min(100.0, max(0.0, initial_soc - 10.0))
        self.ekf.reset(initial_soc=initial_soc / 100.0)
        self.spkf.reset(initial_soc=initial_soc / 100.0)
        self.soh.reset()
        self.history_true_soc.clear()
        self.history_ml_soc.clear()
        self.history_ekf_soc.clear()
        self.history_spkf_soc.clear()
        self.history_cc_soc.clear()
        self.history_cc_wrong_soc.clear()
        self.history_soh.clear()

    def seek(self, sample_idx: int) -> None:
        """Seek simulation to a specific sample index, synchronizing all estimators."""
        self.cell.seek(sample_idx)
        cursor = self.cell.cursor
        if len(self.cell.df) > 0 and cursor < len(self.cell.df):
            row = self.cell.df.iloc[cursor]
            target_soc = float(row.get("soc_true", row.get("soc_ml", 100.0)))
            current_i = float(row.get("I", 0.0))
        else:
            target_soc = 100.0
            current_i = 0.0

        if pd.isna(target_soc):
            target_soc = 100.0

        self.firmware.reset()
        init_state = TP4056State.CHARGING if current_i < -0.05 else TP4056State.DISCHARGING
        self.tp4056.reset(initial_state=init_state)

        self.cc_biased_soc = target_soc
        self.cc_wrong_soc = min(100.0, max(0.0, target_soc - 10.0))
        self.ekf.reset(initial_soc=target_soc / 100.0)
        self.spkf.reset(initial_soc=target_soc / 100.0)
        self.soh.reset()
        
    def step(self) -> Dict[str, Any]:
        """Execute one 5-second sample through the hardware chain."""
        # 1. Physical Cell
        cell_data = self.cell.step()
        v_true = cell_data["v_true"]
        i_true = cell_data["i_true"]
        soc_true = cell_data["soc_true"]
        
        # 2. Resistor Divider (/2)
        div_data = self.divider.step(v_true)
        v_pin = div_data["v_pin"]
        
        # 3. ESP32 ADC
        adc_data = self.adc.step(v_pin)
        v_recon = adc_data["v_cell_reconstructed"]
        adc_code = adc_data["adc_code"]
        
        # 4. Current Sensor
        curr_data = self.current_sensor.step(i_true)
        i_meas = curr_data["i_measured"]
        
        # 5. TP4056 Charger Controller
        tp_data = self.tp4056.step(v_recon, i_meas)
        status = tp_data["status"]
        
        # 6. ESP32 Firmware Execution
        fw_data = self.firmware.step(v_recon, i_meas, status)
        soc_ml = fw_data["soc_estimate"]
        wifi_payload = fw_data["wifi_payload"]
        
        # 7. Reference Estimators
        # Coulomb counting integrates i_meas
        delta_q = i_meas * self.dt_s
        self.cc_biased_soc -= (delta_q / self.capacity_as) * 100.0
        self.cc_wrong_soc -= (delta_q / self.capacity_as) * 100.0
        
        self.cc_biased_soc = float(np.clip(self.cc_biased_soc, 0.0, 100.0))
        self.cc_wrong_soc = float(np.clip(self.cc_wrong_soc, 0.0, 100.0))

        # 8. Kalman Filter Estimators (EKF & SPKF)
        ekf_data = self.ekf.step(v_recon, i_meas)
        spkf_data = self.spkf.step(v_recon, i_meas)
        soc_ekf = ekf_data["soc_ekf"]
        soc_spkf = spkf_data["soc_spkf"]

        # 9. Dual SOH Estimation (Capacity Fade + Resistance Fade)
        soh_data = self.soh.step(v_recon, i_meas, self.dt_s)
        
        # Track history
        self.history_true_soc.append(soc_true)
        self.history_ml_soc.append(soc_ml)
        self.history_ekf_soc.append(soc_ekf)
        self.history_spkf_soc.append(soc_spkf)
        self.history_cc_soc.append(self.cc_biased_soc)
        self.history_cc_wrong_soc.append(self.cc_wrong_soc)
        self.history_soh.append(soh_data)
        
        # Running accuracy metrics
        err_ml = abs(soc_true - soc_ml)
        err_cc = abs(soc_true - self.cc_biased_soc)
        err_ekf = abs(soc_true - soc_ekf)
        err_spkf = abs(soc_true - soc_spkf)
        hist_true = np.array(self.history_true_soc)
        hist_ml = np.array(self.history_ml_soc)
        hist_cc = np.array(self.history_cc_soc)
        hist_ekf = np.array(self.history_ekf_soc)
        hist_spkf = np.array(self.history_spkf_soc)
        running_ml_mae = float(np.mean(np.abs(hist_true - hist_ml)))
        running_ml_rmse = float(np.sqrt(np.mean((hist_true - hist_ml) ** 2)))
        running_cc_mae = float(np.mean(np.abs(hist_true - hist_cc)))
        running_ekf_mae = float(np.mean(np.abs(hist_true - hist_ekf)))
        running_spkf_mae = float(np.mean(np.abs(hist_true - hist_spkf)))
        
        return {
            "time_s": cell_data["time_s"],
            "is_done": cell_data["is_done"],
            # Signals across virtual hardware chain
            "chain": {
                "cell_v": v_true,
                "cell_i": i_true,
                "pin_v": v_pin,
                "adc_code": adc_code,
                "reconstructed_v": v_recon,
                "measured_i": i_meas,
                "v_smooth": fw_data["v_smooth"],
            },
            # Indicators
            "status": status,
            "led_chrg": tp_data["led_chrg"],
            "led_stdby": tp_data["led_stdby"],
            # Estimator states
            "soc_true": soc_true,
            "soc_ml": soc_ml,
            "soc_ekf": soc_ekf,
            "soc_spkf": soc_spkf,
            "soc_cc_biased": self.cc_biased_soc,
            "soc_cc_wrong": self.cc_wrong_soc,
            # State of Health
            "soh": soh_data,
            # Telemetry Wi-Fi Payload (ESP32 output)
            "wifi_payload": wifi_payload,
            "wifi_json": fw_data["wifi_payload_json"],
            # Running Metrics
            "metrics": {
                "running_ml_mae": running_ml_mae,
                "running_ml_rmse": running_ml_rmse,
                "running_cc_mae": running_cc_mae,
                "running_ekf_mae": running_ekf_mae,
                "running_spkf_mae": running_spkf_mae,
                "instant_ml_error": err_ml,
                "instant_cc_error": err_cc,
                "instant_ekf_error": err_ekf,
                "instant_spkf_error": err_spkf,
                "infer_time_us": fw_data.get("infer_time_us", 6.0),
            }
        }


def run_headless_simulation(
    cycle_name: str = "pair10",
    noise_mv: float = 5.0,
    offset_ma: float = 30.0,
    bits: int = 12,
    seed: int = 42,
    print_interval: int = 100,
) -> Dict[str, Any]:
    """Execute complete cycle headless and return evaluation metrics."""
    dataset = load_cached_dataset()
    
    if cycle_name.lower() in ["pair10", "1", "pair1"]:
        df = build_cycle_pair(dataset["Discharge_10.csv"], dataset["Load_10.csv"])
        title = "Pair 10 (1.0 A)"
    elif cycle_name.lower() in ["pair20", "2", "pair2"]:
        df = build_cycle_pair(dataset["Discharge_20.csv"], dataset["Load_20.csv"])
        title = "Pair 20 (2.0 A)"
    elif cycle_name.lower() in ["pair30", "3", "pair3"]:
        df = build_cycle_pair(dataset["Discharge_30.csv"], dataset["Load_30.csv"])
        title = "Pair 30 (3.0 A)"
    elif cycle_name.lower() in ["discharge02", "dis02", "0.5"]:
        df = dataset["Discharge_02.csv"]
        title = "Discharge 02 (0.5 A)"
    else:
        # Search dataset
        matching = [k for k in dataset.keys() if cycle_name.lower() in k.lower()]
        if matching:
            df = dataset[matching[0]]
            title = matching[0]
        else:
            raise ValueError(f"Unknown cycle: {cycle_name}")
            
    twin = DigitalTwin(
        cycle_df=df,
        cycle_name=title,
        adc_noise_mv=noise_mv,
        current_offset_ma=offset_ma,
        adc_bits=bits,
        seed=seed,
    )
    
    print("=" * 70)
    print(f"HEADLESS DIGITAL TWIN SIMULATION: {title}")
    print(f"Params: ADC Noise={noise_mv} mV | Offset={offset_ma} mA | ADC Bits={bits} | Seed={seed}")
    print("=" * 70)
    
    step_count = 0
    while not twin.cell.is_done:
        frame = twin.step()
        step_count += 1
        if step_count % print_interval == 0 or twin.cell.is_done:
            p = frame["wifi_payload"]
            m = frame["metrics"]
            print(f"[{frame['time_s']/3600.0:5.2f}h] Payload: {frame['wifi_json']} | ML MAE: {m['running_ml_mae']:.2f}% | CC MAE: {m['running_cc_mae']:.2f}%")
            
    y_true = np.array(twin.history_true_soc)
    y_ml = np.array(twin.history_ml_soc)
    y_cc = np.array(twin.history_cc_soc)
    
    ml_mae = float(np.mean(np.abs(y_true - y_ml)))
    ml_rmse = float(np.sqrt(np.mean((y_true - y_ml) ** 2)))
    ml_max = float(np.max(np.abs(y_true - y_ml)))
    
    cc_mae = float(np.mean(np.abs(y_true - y_cc)))
    cc_rmse = float(np.sqrt(np.mean((y_true - y_cc) ** 2)))
    cc_max = float(np.max(np.abs(y_true - y_cc)))
    
    print("\n--- FINAL RUN METRICS ---")
    print(f"ML 3-16-1 MAE:     {ml_mae:.2f}%  (RMSE: {ml_rmse:.2f}%, Max: {ml_max:.2f}%)")
    print(f"Coulomb Count MAE: {cc_mae:.2f}%  (RMSE: {cc_rmse:.2f}%, Max: {cc_max:.2f}%)")
    print("Target (< 2-3% error):", "MET ✓" if ml_mae < 3.0 else "MISSED ✗")
    
    return {
        "cycle": title,
        "total_samples": step_count,
        "ml_mae": ml_mae,
        "ml_rmse": ml_rmse,
        "ml_max": ml_max,
        "cc_mae": cc_mae,
        "cc_rmse": cc_rmse,
        "cc_max": cc_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Headless Digital Twin Simulation")
    parser.add_argument("--cycle", type=str, default="pair10", help="Cycle name (pair10, pair20, pair30, dis02)")
    parser.add_argument("--noise", type=float, default=5.0, help="ADC noise sigma in mV (default 5)")
    parser.add_argument("--offset", type=float, default=30.0, help="Current sensor offset in mA (default 30)")
    parser.add_argument("--bits", type=int, default=12, help="ADC resolution in bits (default 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--step-print", type=int, default=400, help="Print interval in steps")
    args = parser.parse_args()
    
    run_headless_simulation(
        cycle_name=args.cycle,
        noise_mv=args.noise,
        offset_ma=args.offset,
        bits=args.bits,
        seed=args.seed,
        print_interval=args.step_print,
    )


if __name__ == "__main__":
    main()
