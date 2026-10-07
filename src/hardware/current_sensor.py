"""Current Sensor Hardware Component (e.g. INA219 / Hall-Effect Sensor).

Simulates current measurement with constant sensor bias/offset and optional noise:
I_measured = I_true + offset_a + noise
"""

from typing import Dict, Any, Optional
import numpy as np


class CurrentSensor:
    """Simulates current shunt / sensor with DC offset and Gaussian noise."""
    
    def __init__(
        self,
        offset_ma: float = 30.0,
        noise_sigma_ma: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.offset_a = float(offset_ma) / 1000.0
        self.noise_sigma_a = float(noise_sigma_ma) / 1000.0
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def step(self, i_true: float) -> Dict[str, Any]:
        """Convert true cell current to measured sensor reading."""
        noise = self.rng.normal(0.0, self.noise_sigma_a) if self.noise_sigma_a > 0.0 else 0.0
        i_measured = i_true + self.offset_a + noise
        return {
            "i_measured": float(i_measured),
            "offset_ma": self.offset_a * 1000.0,
        }
