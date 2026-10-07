"""Extended Kalman Filter (EKF) and Sigma-Point Kalman Filter (SPKF) for Battery SOC Estimation.

Implements physics-based State of Charge estimation using a 1-RC Thevenin
Equivalent Circuit Model (ECM):
- States: x = [SOC, V_p]^T (SOC in [0, 1], polarization voltage in Volts)
- Inputs: Current I (A) (positive for discharge, negative for charge)
- Observations: Terminal cell voltage V_t (V)

Algorithms:
1. Extended Kalman Filter (EKF):
   - Linearizes observation function using analytical Jacobian dOCV/dSOC
   - Standard predict and update cycles with process noise Q and measurement noise R
2. Sigma-Point Kalman Filter (SPKF / Unscented Kalman Filter - UKF):
   - Generates 2L + 1 = 5 deterministic sigma points using scaled unscented transform
   - Propagates sigma points through non-linear OCV curve directly (no Jacobians required)
   - Accurately captures mean and covariance through non-linear voltage transitions
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


# 6th-degree polynomial coefficients fit to 18650 cell OCV-SOC characteristic curve
# Monotonic strictly non-decreasing from 3.00V (0% SOC) to 4.18V (100% SOC)
_OCV_POLY = np.array([-38.147, 132.893, -181.704, 123.614, -43.197, 7.717, 3.003], dtype=np.float64)
_DOCV_POLY = np.polyder(_OCV_POLY)


def ocv_function(soc_normalized: float) -> float:
    """Compute Open-Circuit Voltage (V) from normalized SOC [0.0, 1.0]."""
    s = float(np.clip(soc_normalized, 0.0, 1.0))
    return float(np.polyval(_OCV_POLY, s))


def docv_dsoc_function(soc_normalized: float) -> float:
    """Compute analytical derivative dOCV/dSOC (V/unit) for EKF Jacobian."""
    s = float(np.clip(soc_normalized, 0.0, 1.0))
    return float(np.polyval(_DOCV_POLY, s))


class BatteryECM:
    """1-RC Thevenin Equivalent Circuit Model for 18650 Li-ion cell."""

    def __init__(
        self,
        capacity_ah: float = 2.65,
        r0_ohm: float = 0.055,
        r1_ohm: float = 0.040,
        c1_farad: float = 1500.0,
        dt_s: float = 5.0,
        coulomb_eff: float = 1.0,
    ):
        self.capacity_as = float(capacity_ah) * 3600.0
        self.r0 = float(r0_ohm)
        self.r1 = float(r1_ohm)
        self.c1 = float(c1_farad)
        self.dt = float(dt_s)
        self.eta = float(coulomb_eff)
        self.tau = self.r1 * self.c1
        self.exp_dt_tau = float(np.exp(-self.dt / self.tau))

    def state_transition_matrix(self) -> np.ndarray:
        """Matrix A for x_k = A x_{k-1} + B u_{k-1}."""
        return np.array([
            [1.0, 0.0],
            [0.0, self.exp_dt_tau],
        ], dtype=np.float64)

    def input_matrix(self) -> np.ndarray:
        """Matrix B for current input."""
        b_soc = -(self.eta * self.dt) / self.capacity_as
        b_vp = self.r1 * (1.0 - self.exp_dt_tau)
        return np.array([b_soc, b_vp], dtype=np.float64)


class ExtendedKalmanFilter:
    """Extended Kalman Filter (EKF) for battery SOC estimation."""

    def __init__(
        self,
        initial_soc: float = 1.0, # 0.0 to 1.0
        initial_vp: float = 0.0,
        ecm: Optional[BatteryECM] = None,
        q_soc: float = 1e-6,
        q_vp: float = 1e-5,
        r_meas: float = 4e-3, # Measurement noise variance
    ):
        self.ecm = ecm if ecm is not None else BatteryECM()
        self.x = np.array([float(initial_soc), float(initial_vp)], dtype=np.float64)
        self.P = np.array([
            [0.01, 0.0],
            [0.0, 0.001],
        ], dtype=np.float64)
        self.Q = np.array([
            [float(q_soc), 0.0],
            [0.0, float(q_vp)],
        ], dtype=np.float64)
        self.R = float(r_meas)

    def reset(self, initial_soc: float = 1.0, initial_vp: float = 0.0) -> None:
        """Reset state vector and error covariance."""
        self.x = np.array([float(initial_soc), float(initial_vp)], dtype=np.float64)
        self.P = np.array([
            [0.01, 0.0],
            [0.0, 0.001],
        ], dtype=np.float64)

    def step(self, v_meas: float, i_meas: float) -> Dict[str, float]:
        """Execute one EKF predict-update cycle.
        
        Args:
            v_meas: Measured terminal cell voltage (V)
            i_meas: Measured cell current (A, positive = discharge)
            
        Returns:
            Dictionary with estimated SOC (%), terminal voltage prediction, and Kalman gain
        """
        A = self.ecm.state_transition_matrix()
        B = self.ecm.input_matrix()

        # 1. State Prediction
        x_pred = A @ self.x + B * i_meas
        x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)
        P_pred = A @ self.P @ A.T + self.Q

        # 2. Measurement Prediction
        soc_pred = float(x_pred[0])
        vp_pred = float(x_pred[1])
        ocv_pred = ocv_function(soc_pred)
        v_pred = ocv_pred - vp_pred - i_meas * self.ecm.r0

        # 3. Jacobian Calculation: H = [dOCV/dSOC, -1]
        h_soc = docv_dsoc_function(soc_pred)
        H = np.array([[h_soc, -1.0]], dtype=np.float64)

        # 4. Kalman Gain & Update
        S = float((H @ P_pred @ H.T)[0, 0] + self.R)
        K = (P_pred @ H.T) / max(1e-9, S) # shape (2, 1)

        v_err = float(v_meas - v_pred)
        self.x = x_pred + (K * v_err).flatten()
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        I_mat = np.eye(2, dtype=np.float64)
        self.P = (I_mat - K @ H) @ P_pred

        soc_pct = float(self.x[0] * 100.0)
        return {
            "soc_ekf": round(soc_pct, 2),
            "v_pred_ekf": round(v_pred, 4),
            "vp_ekf": round(float(self.x[1]), 4),
            "kalman_gain_soc": round(float(K[0, 0]), 5),
        }


class SigmaPointKalmanFilter:
    """Sigma-Point Kalman Filter (SPKF / UKF) for battery SOC estimation."""

    def __init__(
        self,
        initial_soc: float = 1.0,
        initial_vp: float = 0.0,
        ecm: Optional[BatteryECM] = None,
        q_soc: float = 1e-6,
        q_vp: float = 1e-5,
        r_meas: float = 4e-3,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 1.0,
    ):
        self.ecm = ecm if ecm is not None else BatteryECM()
        self.L = 2 # State dimension: [SOC, V_p]
        self.x = np.array([float(initial_soc), float(initial_vp)], dtype=np.float64)
        self.P = np.array([
            [0.01, 0.0],
            [0.0, 0.001],
        ], dtype=np.float64)
        self.Q = np.array([
            [float(q_soc), 0.0],
            [0.0, float(q_vp)],
        ], dtype=np.float64)
        self.R = float(r_meas)

        # Julier / Van der Merwe Scaled Unscented Transform weights
        self.lam = (alpha ** 2) * (self.L + kappa) - self.L
        gamma = self.L + self.lam
        self.c = float(np.sqrt(max(1e-9, gamma)))

        # Weight vectors (size 2L + 1 = 5)
        self.Wm = np.zeros(2 * self.L + 1, dtype=np.float64)
        self.Wc = np.zeros(2 * self.L + 1, dtype=np.float64)

        self.Wm[0] = self.lam / gamma
        self.Wc[0] = self.lam / gamma + (1.0 - alpha ** 2 + beta)
        for i in range(1, 2 * self.L + 1):
            self.Wm[i] = 1.0 / (2.0 * gamma)
            self.Wc[i] = 1.0 / (2.0 * gamma)

    def reset(self, initial_soc: float = 1.0, initial_vp: float = 0.0) -> None:
        """Reset state vector and error covariance."""
        self.x = np.array([float(initial_soc), float(initial_vp)], dtype=np.float64)
        self.P = np.array([
            [0.01, 0.0],
            [0.0, 0.001],
        ], dtype=np.float64)

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate 5 deterministic sigma points from x and P."""
        # Ensure positive-definite covariance matrix
        P_sym = 0.5 * (self.P + self.P.T) + np.eye(self.L) * 1e-9
        try:
            L_mat = np.linalg.cholesky(P_sym)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if slightly non-positive
            vals, vecs = np.linalg.eigh(P_sym)
            vals = np.maximum(vals, 1e-9)
            L_mat = vecs @ np.diag(np.sqrt(vals))

        sigma_points = np.zeros((2 * self.L + 1, self.L), dtype=np.float64)
        sigma_points[0] = self.x

        for i in range(self.L):
            col = self.c * L_mat[:, i]
            sigma_points[1 + i] = self.x + col
            sigma_points[1 + self.L + i] = self.x - col

        # Bound SOC component to [0, 1]
        sigma_points[:, 0] = np.clip(sigma_points[:, 0], 0.0, 1.0)
        return sigma_points

    def step(self, v_meas: float, i_meas: float) -> Dict[str, float]:
        """Execute one SPKF predict-update cycle with unscented transform.
        
        Args:
            v_meas: Measured terminal cell voltage (V)
            i_meas: Measured cell current (A)
            
        Returns:
            Dictionary with estimated SOC (%), terminal voltage prediction, and residual
        """
        A = self.ecm.state_transition_matrix()
        B = self.ecm.input_matrix()

        # 1. Generate prior sigma points
        sigmas = self._generate_sigma_points()

        # 2. Time update: propagate sigma points through state dynamics
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(2 * self.L + 1):
            x_next = A @ sigmas[i] + B * i_meas
            x_next[0] = np.clip(x_next[0], 0.0, 1.0)
            sigmas_pred[i] = x_next

        # Predicted state mean
        x_pred = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)
        x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)

        # Predicted state covariance
        P_pred = np.zeros((self.L, self.L), dtype=np.float64)
        for i in range(2 * self.L + 1):
            diff = sigmas_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += self.Q

        # 3. Measurement update: propagate sigma points through non-linear OCV observation
        gamma_y = np.zeros(2 * self.L + 1, dtype=np.float64)
        for i in range(2 * self.L + 1):
            soc_pt = float(sigmas_pred[i, 0])
            vp_pt = float(sigmas_pred[i, 1])
            ocv_pt = ocv_function(soc_pt)
            gamma_y[i] = ocv_pt - vp_pt - i_meas * self.ecm.r0

        # Predicted measurement mean
        y_pred = float(np.sum(self.Wm * gamma_y))

        # Innovation and cross covariance
        P_yy = float(np.sum(self.Wc * ((gamma_y - y_pred) ** 2)) + self.R)
        P_xy = np.zeros(self.L, dtype=np.float64)
        for i in range(2 * self.L + 1):
            diff_x = sigmas_pred[i] - x_pred
            diff_y = gamma_y[i] - y_pred
            P_xy += self.Wc[i] * diff_x * diff_y

        # 4. Kalman Gain & State Update
        K = P_xy / max(1e-9, P_yy) # shape (2,)
        residual = float(v_meas - y_pred)

        self.x = x_pred + K * residual
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = P_pred - np.outer(K, K) * P_yy

        soc_pct = float(self.x[0] * 100.0)
        return {
            "soc_spkf": round(soc_pct, 2),
            "v_pred_spkf": round(y_pred, 4),
            "vp_spkf": round(float(self.x[1]), 4),
            "residual_spkf": round(residual, 4),
        }
