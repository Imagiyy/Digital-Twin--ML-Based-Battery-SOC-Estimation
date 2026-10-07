"""Voltage Divider Component.

Steps down cell terminal voltage (up to 4.2V) by ratio 0.5 to fit
within ESP32 ADC safe input range (0 - 3.3V).
Vpin = V_cell * (ratio * (1 + tolerance_gain_error))
"""

from typing import Dict, Any


class VoltageDivider:
    """Simulates 1:2 resistor voltage divider (e.g. two matched 100k ohm resistors)."""
    
    def __init__(self, ratio: float = 0.5, tolerance_gain_error: float = 0.0):
        self.ratio = ratio
        self.gain = ratio * (1.0 + tolerance_gain_error)
        
    def step(self, v_cell: float) -> Dict[str, Any]:
        """Convert cell voltage to ADC pin voltage."""
        v_pin = v_cell * self.gain
        return {
            "v_pin": v_pin,
            "divider_ratio": self.ratio,
        }
