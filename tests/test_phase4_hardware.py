"""Unit tests for Phase 4: Virtual Hardware Twin Components and Signals."""

import numpy as np
import pytest

from src.hardware.divider import VoltageDivider
from src.hardware.adc import ESP32ADC
from src.hardware.current_sensor import CurrentSensor
from src.hardware.tp4056 import TP4056, TP4056State
from src.hardware.esp32 import Esp32Firmware
from src.mlp_infer import PureNumpyMLP
from src.twin import DigitalTwin


def test_voltage_divider_ratio():
    """Verify voltage divider steps down cell voltage by 0.5."""
    divider = VoltageDivider(ratio=0.5, tolerance_gain_error=0.0)
    out = divider.step(4.0)
    assert out["v_pin"] == pytest.approx(2.0, abs=1e-5)


def test_esp32_adc_quantisation_and_reference_code():
    """Verify 12-bit ADC mapping, including the Master Prompt spec check:
    1.956 V -> code about 2429.
    """
    # Disable random noise to test deterministic quantisation
    adc = ESP32ADC(vref=3.3, bits=12, noise_sigma_mv=0.0, divider_ratio=0.5)
    
    # Check 1.956 V input: 1.956 / 3.3 * 4095 = 2427.27 -> rounds to 2427 (about 2429)
    out = adc.step(1.956)
    code = out["adc_code"]
    assert abs(code - 2429) <= 2, f"Expected code about 2429, got {code}"
    
    # Check rail limits
    assert adc.step(0.0)["adc_code"] == 0
    assert adc.step(3.3)["adc_code"] == 4095
    assert adc.step(3.5)["adc_code"] == 4095 # clipped to rail


def test_current_sensor_offset():
    """Verify current sensor applies 30 mA offset."""
    sensor = CurrentSensor(offset_ma=30.0, noise_sigma_ma=0.0)
    out = sensor.step(1.0)
    assert out["i_measured"] == pytest.approx(1.030, abs=1e-5)
    
    out_chg = sensor.step(-1.0)
    assert out_chg["i_measured"] == pytest.approx(-0.970, abs=1e-5)


def test_tp4056_state_machine_and_leds():
    """Verify TP4056 states and LED signals under charge, discharge, and CV termination."""
    tp = TP4056(cv_threshold_v=4.18, termination_current_a=0.10, debounce_samples=1)
    
    # 1. Discharging into load (+1.0 A)
    res = tp.step(v_cell=3.8, i_cell=1.0)
    assert res["status"] == TP4056State.DISCHARGING.value
    assert res["led_chrg"] is False
    assert res["led_stdby"] is False
    
    # 2. Charging (-1.0 A) below CV threshold
    res = tp.step(v_cell=3.9, i_cell=-1.0)
    assert res["status"] == TP4056State.CHARGING.value
    assert res["led_chrg"] is True
    assert res["led_stdby"] is False
    
    # 3. CV termination: V >= 4.18 V and current decayed to -0.05 A
    res = tp.step(v_cell=4.20, i_cell=-0.05)
    assert res["status"] == TP4056State.CHARGED_STANDBY.value
    assert res["led_chrg"] is False
    assert res["led_stdby"] is True

    # 4. NaN input immunity (holds state safely)
    res_nan = tp.step(v_cell=float("nan"), i_cell=float("nan"))
    assert res_nan["status"] == TP4056State.CHARGED_STANDBY.value
    assert res_nan["led_stdby"] is True

    # 5. Reset functionality
    tp.reset(initial_state=TP4056State.DISCHARGING)
    assert tp.current_state == TP4056State.DISCHARGING
    assert tp._debounce_count == 0


def test_wifi_payload_structure():
    """Verify standard Wi-Fi payload schema emitted by virtual firmware."""
    model = PureNumpyMLP()
    firmware = Esp32Firmware(model, window_size=12)
    
    out = firmware.step(v_recon=3.915, i_measured=-1.02, charger_status="CHARGING")
    payload = out["wifi_payload"]
    
    assert "v" in payload and "i" in payload and "soc" in payload and "status" in payload
    assert payload["v"] == 3.915
    assert payload["i"] == -1.02
    assert isinstance(payload["soc"], float)
    assert payload["status"] == "CHARGING"
