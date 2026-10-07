"""Tests for SOH (Capacity & Resistance Fade), Kalman Filters (EKF & SPKF), and API Endpoints."""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.soh import (
    CapacityFadeEstimator,
    ResistanceFadeEstimator,
    CombinedSOHEstimator,
    compute_nasa_cycle_degradation_history,
)
from src.kalman import (
    BatteryECM,
    ExtendedKalmanFilter,
    SigmaPointKalmanFilter,
    ocv_function,
    docv_dsoc_function,
)
from src.server import app

client = TestClient(app)


# ============================================================================
# SOH Estimator Tests
# ============================================================================

def test_capacity_fade_estimator():
    """Verify Capacity Fade algorithm (SOH_C) tracks Ah throughput and clips properly."""
    cap_est = CapacityFadeEstimator(nominal_capacity_ah=2.65, alpha_fade=0.0035)
    assert cap_est.actual_capacity_ah == pytest.approx(2.65)
    assert cap_est.soh_c_pct == pytest.approx(100.0)

    # Step discharge 2.0 A for 1800 s (0.5 h = 1.0 Ah throughput)
    res = cap_est.step(current_a=2.0, dt_s=1800.0)
    assert res["throughput_ah"] == pytest.approx(1.0, rel=1e-3)
    assert res["soh_capacity_pct"] < 100.0
    assert res["soh_capacity_pct"] > 95.0
    assert 0.0 <= res["soh_capacity_pct"] <= 100.0

    # Test explicit capacity update
    cap_est.set_measured_capacity(2.12)  # exactly 80% of 2.65 Ah
    assert cap_est.soh_c_pct == pytest.approx(80.0, rel=1e-2)


def test_resistance_fade_estimator():
    """Verify Resistance Fade algorithm (SOH_R) extracts dynamic R0 on load transitions."""
    res_est = ResistanceFadeEstimator(r_fresh_ohm=0.055, r_eol_ohm=0.110, filter_gain=0.1)
    assert res_est.soh_r_pct == pytest.approx(100.0)

    # Initial steady point
    res_est.step(v_meas=4.00, i_meas=0.0)

    # Load step: current steps from 0 to 2 A, voltage drops from 4.00V to 3.86V (delta V = 0.14V, delta I = 2A -> R0 = 0.070 Ohm)
    res1 = res_est.step(v_meas=3.86, i_meas=2.0)
    assert res1["estimated_r0_mohm"] > 55.0  # R0 increased toward 70 mOhm
    assert res1["soh_resistance_pct"] < 100.0
    assert 0.0 <= res1["soh_resistance_pct"] <= 100.0

    # Directly set cycle resistance to EOL resistance (110 mOhm)
    res_est.set_cycle_resistance(0.110)
    assert res_est.soh_r_pct == pytest.approx(0.0, abs=0.1)


def test_combined_soh_estimator():
    """Verify Combined SOH combines SOH_C and SOH_R with correct weights and status."""
    comb = CombinedSOHEstimator(nominal_capacity_ah=2.65, r_fresh_ohm=0.055, r_eol_ohm=0.110, weight_capacity=0.6, weight_resistance=0.4, cycle_number=0)
    res = comb.get_soh()
    assert res["soh_overall_pct"] == pytest.approx(100.0)
    assert res["status"] == "EXCELLENT"
    assert res["rul_cycles_est"] > 0

    # Inject aged state
    comb.cap_estimator.set_measured_capacity(2.25)  # ~85%
    comb.res_estimator.set_cycle_resistance(0.075)   # ~63.6%
    aged_res = comb.get_soh()
    expected_soh = 0.6 * (2.25 / 2.65 * 100.0) + 0.4 * ((0.110 - 0.075) / (0.110 - 0.055) * 100.0)
    assert aged_res["soh_overall_pct"] == pytest.approx(expected_soh, rel=1e-2)
    assert aged_res["status"] in ["GOOD", "MODERATE AGING", "END OF LIFE (REPLACE)"]


def test_nasa_cycle_degradation_history():
    """Verify NASA 30-cycle degradation history computation."""
    history = compute_nasa_cycle_degradation_history()
    assert len(history) == 30
    first_cyc = history[0]
    last_cyc = history[-1]
    assert first_cyc["cycle"] == 1
    assert last_cyc["cycle"] == 30
    assert first_cyc["soh_capacity_pct"] > last_cyc["soh_capacity_pct"]
    assert first_cyc["r0_mohm"] < last_cyc["r0_mohm"]


# ============================================================================
# Kalman Filter Tests (EKF & SPKF)
# ============================================================================

def test_ocv_function_properties():
    """Verify OCV polynomial is monotonic and bounded between 3.0V and 4.2V."""
    v_0 = ocv_function(0.0)
    v_50 = ocv_function(0.5)
    v_100 = ocv_function(1.0)
    assert 2.9 <= v_0 <= 3.2
    assert 3.6 <= v_50 <= 3.8
    assert 4.1 <= v_100 <= 4.25
    assert v_0 < v_50 < v_100

    # Derivative dOCV/dSOC should be strictly positive across operating range
    for s in np.linspace(0.1, 0.9, 9):
        docv = docv_dsoc_function(float(s))
        assert docv > 0.0


def test_extended_kalman_filter():
    """Verify Extended Kalman Filter step execution and output bounds."""
    ekf = ExtendedKalmanFilter(initial_soc=1.0)
    step1 = ekf.step(v_meas=4.15, i_meas=1.0)
    assert "soc_ekf" in step1
    assert "v_pred_ekf" in step1
    assert 0.0 <= step1["soc_ekf"] <= 100.0
    assert 3.0 <= step1["v_pred_ekf"] <= 4.3

    # Consecutive steps with low voltage should drive estimated SOC down
    for _ in range(50):
        step_n = ekf.step(v_meas=3.50, i_meas=2.0)
    assert step_n["soc_ekf"] < 100.0


def test_sigma_point_kalman_filter():
    """Verify Sigma-Point Kalman Filter (SPKF / UKF) generates valid sigma points and updates."""
    spkf = SigmaPointKalmanFilter(initial_soc=1.0)
    step1 = spkf.step(v_meas=4.15, i_meas=1.0)
    assert "soc_spkf" in step1
    assert "v_pred_spkf" in step1
    assert 0.0 <= step1["soc_spkf"] <= 100.0
    assert 3.0 <= step1["v_pred_spkf"] <= 4.3

    for _ in range(50):
        step_n = spkf.step(v_meas=3.50, i_meas=2.0)
    assert step_n["soc_spkf"] < 100.0


# ============================================================================
# API Endpoint Tests
# ============================================================================

def test_api_soh_endpoint():
    """Verify GET /api/soh returns structured dual-fade SOH data."""
    response = client.get("/api/soh")
    assert response.status_code == 200
    data = response.json()
    assert "current_soh" in data
    assert "degradation_history" in data
    assert len(data["degradation_history"]) == 30
    assert "soh_combined_pct" in data["current_soh"] or "soh_capacity_pct" in data["current_soh"]


def test_api_algos_endpoint():
    """Verify GET /api/algos returns algorithm comparison and recommendation verdict."""
    response = client.get("/api/algos")
    assert response.status_code == 200
    data = response.json()
    assert "recommendation" in data
    assert "scorecard" in data
    assert data["recommendation"]["best_overall"] == "MLP 3-16-1 (Proposed Neural Network)"
    assert len(data["scorecard"]) >= 5
    # Verify rankings
    ranks = [a["rank"] for a in data["scorecard"]]
    assert 1 in ranks
    assert 2 in ranks


def test_api_metrics_includes_filters():
    """Verify GET /api/metrics includes SPKF and EKF MAE tracking."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "ml_mae" in data
    assert "ekf_mae" in data
    assert "spkf_mae" in data
