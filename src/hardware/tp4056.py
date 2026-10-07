"""TP4056 Li-Ion Linear Charger State Machine Component.

Models the physical TP4056 charge management IC:
- States: CHARGING, DISCHARGING, CHARGED_STANDBY
- Pin indicators:
    * CHRG LED: True while charging (Red pin active-low LED lit)
    * STDBY LED: True when charge cycle complete / standby (Green/Blue LED lit)
    * DISCHARGING: Both LEDs False (unpowered / passive discharge)
- Transitions based on:
    * Current direction (negative = charging into cell, positive = discharge into load)
    * CV termination criteria: V >= 4.18V and |I| < 0.10A (C/10 cutoff)
    * Debounce counter preventing chatter near transition thresholds
"""

import math
from enum import Enum
from typing import Dict, Any


class TP4056State(str, Enum):
    CHARGING = "CHARGING"
    DISCHARGING = "DISCHARGING"
    CHARGED_STANDBY = "CHARGED_STANDBY"


class TP4056:
    """Simulates TP4056 charger controller state machine with LED outputs."""
    
    def __init__(
        self,
        cv_threshold_v: float = 4.18,
        termination_current_a: float = 0.10,
        current_threshold_a: float = 0.05,
        debounce_samples: int = 1,
    ):
        self.cv_threshold_v = float(cv_threshold_v)
        self.termination_current_a = float(termination_current_a)
        self.current_threshold_a = float(current_threshold_a)
        self.debounce_samples = int(debounce_samples)
        
        self.current_state = TP4056State.DISCHARGING
        self._target_state = TP4056State.DISCHARGING
        self._debounce_count = 0
        self._prev_soc: Optional[float] = None

    def reset(
        self,
        initial_state: TP4056State = TP4056State.DISCHARGING,
        initial_soc: Optional[float] = None,
    ) -> None:
        """Reset internal state machine and debounce counters."""
        self.current_state = initial_state
        self._target_state = initial_state
        self._debounce_count = 0
        self._prev_soc = float(initial_soc) if initial_soc is not None and not math.isnan(initial_soc) else None
        
    def step(self, v_cell: float, i_cell: float, soc: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate state machine transitions on voltage, current, and SOC samples."""
        if v_cell is None or i_cell is None or math.isnan(v_cell) or math.isnan(i_cell):
            return {
                "status": self.current_state.value,
                "led_chrg": (self.current_state == TP4056State.CHARGING),
                "led_stdby": (self.current_state == TP4056State.CHARGED_STANDBY),
            }

        # Track SOC rate-of-change if SOC is supplied
        d_soc = None
        if soc is not None and not math.isnan(soc):
            if self._prev_soc is not None and not math.isnan(self._prev_soc):
                d_soc = float(soc - self._prev_soc)
            self._prev_soc = float(soc)

        # Primary rule:
        # When SOC decreases (d_soc < -0.005) -> DISCHARGING
        # When SOC increases (d_soc > +0.005) -> CHARGING
        if d_soc is not None and d_soc < -0.005:
            raw_next_state = TP4056State.DISCHARGING
        elif d_soc is not None and d_soc > 0.005:
            if v_cell >= self.cv_threshold_v and abs(i_cell) <= self.termination_current_a and (soc is not None and soc >= 98.0):
                raw_next_state = TP4056State.CHARGED_STANDBY
            else:
                raw_next_state = TP4056State.CHARGING
        elif i_cell > self.current_threshold_a:
            raw_next_state = TP4056State.DISCHARGING
        elif i_cell < -self.current_threshold_a:
            # Check CV termination: cell voltage near 4.2V and current has tapered below C/10
            if v_cell >= self.cv_threshold_v and abs(i_cell) <= self.termination_current_a:
                raw_next_state = TP4056State.CHARGED_STANDBY
            else:
                raw_next_state = TP4056State.CHARGING
        else:
            # Low / zero current (|I| <= 0.05 A) and flat SOC:
            # If cell voltage is full (>= 4.15V) or maintaining charged standby (>= 4.10V) or full SOC (>= 98.0%)
            if v_cell >= 4.15 or (soc is not None and soc >= 98.0) or (self.current_state == TP4056State.CHARGED_STANDBY and v_cell >= 4.10):
                raw_next_state = TP4056State.CHARGED_STANDBY
            else:
                raw_next_state = TP4056State.DISCHARGING
            
        # Hysteresis / Debounce logic
        if self.debounce_samples <= 1:
            self.current_state = raw_next_state
            self._target_state = raw_next_state
            self._debounce_count = 0
        elif raw_next_state != self.current_state:
            if raw_next_state == self._target_state:
                self._debounce_count += 1
                if self._debounce_count >= self.debounce_samples:
                    self.current_state = raw_next_state
                    self._debounce_count = 0
            else:
                self._target_state = raw_next_state
                self._debounce_count = 1
        else:
            self._target_state = self.current_state
            self._debounce_count = 0
            
        # Physical LED indicators
        led_chrg = (self.current_state == TP4056State.CHARGING)
        led_stdby = (self.current_state == TP4056State.CHARGED_STANDBY)
        
        return {
            "status": self.current_state.value,
            "led_chrg": led_chrg,
            "led_stdby": led_stdby,
        }
