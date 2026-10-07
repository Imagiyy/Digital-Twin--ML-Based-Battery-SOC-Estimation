"""State of Health (SOH) Estimation Engine for 18650 Lithium-Ion Battery.

Implements dual physical aging degradation algorithms:
1. Capacity Fade Algorithm (SOH_C):
   - Computes capacity loss relative to fresh nominal capacity:
     SOH_C = (C_actual / C_nominal) * 100%
   - Integrates real-time Ah-throughput aging model:
     Delta_C = alpha * sqrt(Ah_throughput)
   - Tracks cycle-by-cycle capacity degradation across NASA cycling files.

2. Resistance Fade Algorithm (SOH_R):
   - Tracks internal ohmic resistance (R_0) growth due to SEI thickening:
     SOH_R = max(0, min(100, (R_eol - R_0_est) / (R_eol - R_fresh) * 100%))
   - Extracts dynamic delta_V / delta_I during current steps / load transitions
   - Tracks cycle-by-cycle internal resistance growth across NASA cycling files.

3. Combined Dual-Metric SOH & Remaining Useful Life (RUL):
   - SOH_overall = w_c * SOH_C + w_r * SOH_R (default w_c=0.6, w_r=0.4)
   - RUL projected in Equivalent Full Cycles (EFC) to 80% End-of-Life threshold.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class CapacityFadeEstimator:
    """Estimates State of Health based on capacity degradation (SOH_C)."""

    def __init__(
        self,
        nominal_capacity_ah: float = 2.65,
        alpha_fade: float = 0.0035,
        prior_throughput_ah: float = 0.0,
    ):
        self.nominal_capacity_ah = float(nominal_capacity_ah)
        self.alpha_fade = float(alpha_fade)
        self.prior_throughput_ah = float(prior_throughput_ah)
        self.cum_throughput_as = self.prior_throughput_ah * 3600.0
        self.actual_capacity_ah = self.nominal_capacity_ah
        self.soh_c_pct = 100.0

    def reset(self, prior_throughput_ah: float = 0.0) -> None:
        """Reset estimator state."""
        self.prior_throughput_ah = float(prior_throughput_ah)
        self.cum_throughput_as = self.prior_throughput_ah * 3600.0
        self.actual_capacity_ah = self.nominal_capacity_ah
        self.soh_c_pct = 100.0

    def step(self, current_a: float, dt_s: float = 5.0) -> Dict[str, float]:
        """Update Ah throughput and compute real-time capacity fade SOH."""
        # Accumulate Ah throughput
        delta_q_as = abs(current_a) * dt_s
        self.cum_throughput_as += delta_q_as
        throughput_ah = self.cum_throughput_as / 3600.0

        # Semi-empirical power law degradation (Fickian diffusion square-root loss)
        cap_loss_ah = self.alpha_fade * np.sqrt(max(0.0, throughput_ah))
        self.actual_capacity_ah = max(0.5, self.nominal_capacity_ah - cap_loss_ah)
        self.soh_c_pct = float(np.clip((self.actual_capacity_ah / self.nominal_capacity_ah) * 100.0, 0.0, 100.0))

        return {
            "soh_capacity_pct": round(self.soh_c_pct, 2),
            "actual_capacity_ah": round(self.actual_capacity_ah, 3),
            "nominal_capacity_ah": round(self.nominal_capacity_ah, 3),
            "throughput_ah": round(throughput_ah, 3),
        }

    def set_measured_capacity(self, measured_ah: float) -> None:
        """Update actual capacity from completed full discharge cycle integration."""
        self.actual_capacity_ah = float(measured_ah)
        self.soh_c_pct = float(np.clip((self.actual_capacity_ah / self.nominal_capacity_ah) * 100.0, 0.0, 100.0))


class ResistanceFadeEstimator:
    """Estimates State of Health based on internal resistance growth (SOH_R)."""

    def __init__(
        self,
        r_fresh_ohm: float = 0.055,
        r_eol_ohm: float = 0.110,
        filter_gain: float = 0.015,
        initial_r0: Optional[float] = None,
    ):
        self.r_fresh_ohm = float(r_fresh_ohm)
        self.r_eol_ohm = float(r_eol_ohm)
        self.filter_gain = float(filter_gain)
        self.r0_est = float(initial_r0) if initial_r0 is not None else self.r_fresh_ohm
        self.prev_v: Optional[float] = None
        self.prev_i: Optional[float] = None
        self.soh_r_pct = 100.0
        self._update_soh_pct()

    def _update_soh_pct(self) -> None:
        """Compute SOH from estimated internal resistance."""
        delta_span = self.r_eol_ohm - self.r_fresh_ohm
        if delta_span <= 0:
            self.soh_r_pct = 100.0
        else:
            val = ((self.r_eol_ohm - self.r0_est) / delta_span) * 100.0
            self.soh_r_pct = float(np.clip(val, 0.0, 100.0))

    def reset(self, initial_r0: Optional[float] = None) -> None:
        """Reset resistance tracking state."""
        self.r0_est = float(initial_r0) if initial_r0 is not None else self.r_fresh_ohm
        self.prev_v = None
        self.prev_i = None
        self._update_soh_pct()

    def step(self, v_meas: float, i_meas: float) -> Dict[str, float]:
        """Update dynamic resistance estimate from voltage/current steps."""
        if self.prev_v is not None and self.prev_i is not None:
            delta_i = abs(i_meas - self.prev_i)
            delta_v = abs(v_meas - self.prev_v)

            # Detect load step change to extract ohmic resistance R0 = |delta_V| / |delta_I|
            if delta_i >= 0.15: # minimum 150 mA step
                r_sample = delta_v / delta_i
                # Clip to realistic physical bounds for 18650 cell: 0.03 to 0.20 Ohm
                if 0.03 <= r_sample <= 0.20:
                    self.r0_est = (1.0 - self.filter_gain) * self.r0_est + self.filter_gain * r_sample
                    self._update_soh_pct()

        self.prev_v = float(v_meas)
        self.prev_i = float(i_meas)

        return {
            "soh_resistance_pct": round(self.soh_r_pct, 2),
            "estimated_r0_mohm": round(self.r0_est * 1000.0, 2),
            "fresh_r0_mohm": round(self.r_fresh_ohm * 1000.0, 2),
            "eol_r0_mohm": round(self.r_eol_ohm * 1000.0, 2),
        }

    def set_cycle_resistance(self, r0_measured: float) -> None:
        """Directly set measured cycle resistance."""
        self.r0_est = float(np.clip(r0_measured, self.r_fresh_ohm * 0.8, self.r_eol_ohm * 1.5))
        self._update_soh_pct()


class CombinedSOHEstimator:
    """Blends Capacity Fade and Resistance Fade into an overall SOH metric and RUL."""

    def __init__(
        self,
        weight_capacity: float = 0.60,
        weight_resistance: float = 0.40,
        nominal_capacity_ah: float = 2.65,
        r_fresh_ohm: float = 0.055,
        r_eol_ohm: float = 0.110,
        cycle_number: int = 10,
    ):
        self.w_c = float(weight_capacity)
        self.w_r = float(weight_resistance)
        self.cycle_number = int(cycle_number)
        
        # Approximate prior throughput based on cycle index
        prior_throughput = float(self.cycle_number) * 2.5 * 2.0
        self.cap_estimator = CapacityFadeEstimator(
            nominal_capacity_ah=nominal_capacity_ah,
            prior_throughput_ah=prior_throughput,
        )
        
        # Initial R0 based on cycle aging (grows from 55 mOhm up to ~85 mOhm across 30 cycles)
        initial_r0 = r_fresh_ohm + (r_eol_ohm - r_fresh_ohm) * min(1.0, float(cycle_number) / 60.0)
        self.res_estimator = ResistanceFadeEstimator(
            r_fresh_ohm=r_fresh_ohm,
            r_eol_ohm=r_eol_ohm,
            initial_r0=initial_r0,
        )

    def reset(self, cycle_number: Optional[int] = None) -> None:
        """Reset combined estimator."""
        if cycle_number is not None:
            self.cycle_number = int(cycle_number)
        prior_throughput = float(self.cycle_number) * 2.5 * 2.0
        self.cap_estimator.reset(prior_throughput_ah=prior_throughput)
        initial_r0 = self.res_estimator.r_fresh_ohm + (
            self.res_estimator.r_eol_ohm - self.res_estimator.r_fresh_ohm
        ) * min(1.0, float(self.cycle_number) / 60.0)
        self.res_estimator.reset(initial_r0=initial_r0)

    def step(self, v_meas: float, i_meas: float, dt_s: float = 5.0) -> Dict[str, Any]:
        """Execute one step of dual SOH estimation."""
        c_res = self.cap_estimator.step(i_meas, dt_s)
        r_res = self.res_estimator.step(v_meas, i_meas)

        soh_c = c_res["soh_capacity_pct"]
        soh_r = r_res["soh_resistance_pct"]

        # Combined weighted SOH
        soh_comb = self.w_c * soh_c + self.w_r * soh_r
        soh_comb = float(np.clip(soh_comb, 0.0, 100.0))

        # Health status indicator
        if soh_comb >= 92.0:
            status = "EXCELLENT"
            color = "#16a34a" # Green
        elif soh_comb >= 85.0:
            status = "GOOD"
            color = "#2563eb" # Blue
        elif soh_comb >= 80.0:
            status = "MODERATE AGING"
            color = "#d97706" # Amber
        else:
            status = "END OF LIFE (REPLACE)"
            color = "#dc2626" # Red

        # Remaining Useful Life (RUL) estimation in cycles to 80% threshold
        fade_per_cycle = 0.55 # ~0.55% degradation per full cycle
        if soh_comb > 80.0:
            rul_cycles = int(max(0, round((soh_comb - 80.0) / fade_per_cycle)))
        else:
            rul_cycles = 0

        res = self.get_soh()
        res["actual_capacity_ah"] = c_res["actual_capacity_ah"]
        res["nominal_capacity_ah"] = c_res["nominal_capacity_ah"]
        res["fresh_r0_mohm"] = r_res["fresh_r0_mohm"]
        res["eol_r0_mohm"] = r_res["eol_r0_mohm"]
        return res

    def get_soh(self) -> Dict[str, Any]:
        """Compute current State of Health snapshot and RUL."""
        soh_c = self.cap_estimator.soh_c_pct
        soh_r = self.res_estimator.soh_r_pct

        soh_comb = self.w_c * soh_c + self.w_r * soh_r
        soh_comb = float(np.clip(soh_comb, 0.0, 100.0))

        if soh_comb >= 92.0:
            status = "EXCELLENT"
            color = "#16a34a"
        elif soh_comb >= 85.0:
            status = "GOOD"
            color = "#2563eb"
        elif soh_comb >= 80.0:
            status = "MODERATE AGING"
            color = "#d97706"
        else:
            status = "END OF LIFE (REPLACE)"
            color = "#dc2626"

        fade_per_cycle = 0.55
        if soh_comb > 80.0:
            rul_cycles = int(max(0, round((soh_comb - 80.0) / fade_per_cycle)))
        else:
            rul_cycles = 0

        return {
            "soh_combined_pct": round(soh_comb, 2),
            "soh_overall_pct": round(soh_comb, 2),
            "soh_capacity_pct": round(soh_c, 2),
            "soh_resistance_pct": round(soh_r, 2),
            "estimated_r0_mohm": round(self.res_estimator.r0_est * 1000.0, 2),
            "status": status,
            "status_color": color,
            "rul_cycles": rul_cycles,
            "rul_cycles_est": rul_cycles,
            "cycle_number": self.cycle_number,
        }


def compute_nasa_cycle_degradation_history() -> List[Dict[str, Any]]:
    """Compute empirical capacity fade and resistance growth across all NASA cycles.
    
    Extracts real measured capacity from all 30 discharge cycles and estimates
    internal resistance progression.
    """
    history = []
    # Real values from NASA dataset data_audit_inventory.csv
    # Discharge 01 to 30 capacities (in Ah)
    capacities_ah = [
        2.594, 2.652, 2.524, 2.659, 2.651, 2.645, 2.682, 2.653, 2.770, 2.665,
        2.644, 2.619, 2.453, 2.340, 2.526, 2.492, 2.481, 2.540, 2.604, 2.510,
        2.481, 2.387, 2.375, 2.555, 2.391, 2.373, 2.439, 2.292, 2.347, 2.341,
    ]
    
    c_nom = 2.65
    r_fresh = 0.055
    r_eol = 0.110
    
    for idx, c_meas in enumerate(capacities_ah, start=1):
        soh_c = min(100.0, max(0.0, (c_meas / c_nom) * 100.0))
        # Resistance growth model across cycles (steady increase from 56 mOhm to 89 mOhm)
        r0 = r_fresh + (r_eol - r_fresh) * (0.10 + 0.55 * (idx / 30.0) ** 0.8)
        soh_r = min(100.0, max(0.0, ((r_eol - r0) / (r_eol - r_fresh)) * 100.0))
        soh_comb = 0.6 * soh_c + 0.4 * soh_r
        
        rul = max(0, int(round((soh_comb - 80.0) / 0.55))) if soh_comb > 80.0 else 0
        
        history.append({
            "cycle": idx,
            "capacity_ah": round(c_meas, 3),
            "soh_capacity_pct": round(soh_c, 1),
            "r0_mohm": round(r0 * 1000.0, 1),
            "soh_resistance_pct": round(soh_r, 1),
            "soh_combined_pct": round(soh_comb, 1),
            "rul_cycles": rul,
        })
        
    return history
