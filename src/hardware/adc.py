"""ESP32 12-Bit Analog-to-Digital Converter Component.

Simulates the Successive Approximation Register (SAR) ADC on the ESP32:
- Resolution: 12-bit (codes 0 - 4095)
- Reference Voltage: 3.3 V
- Noise: Gaussian noise injected on pin (default 5 mV RMS)
- Quantisation: clips to [0, Vref] and rounds to discrete code
- Reconstructs cell voltage: V_recon = code / 4095 * Vref / ratio
"""

from typing import Dict, Any, Optional
import numpy as np


class ESP32ADC:
    """Simulates the ESP32 12-bit ADC peripheral."""
    
    def __init__(
        self,
        vref: float = 3.3,
        bits: int = 12,
        noise_sigma_mv: float = 5.0,
        divider_ratio: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.vref = float(vref)
        self.bits = int(bits)
        self.max_code = (1 << self.bits) - 1 # 4095 for 12-bit
        self.noise_sigma_v = float(noise_sigma_mv) / 1000.0
        self.divider_ratio = float(divider_ratio)
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def step(self, v_pin: float) -> Dict[str, Any]:
        """Convert input pin voltage into noisy digital code and reconstruct cell voltage."""
        # Add thermal/ADC analog noise
        noise = self.rng.normal(0.0, self.noise_sigma_v) if self.noise_sigma_v > 0.0 else 0.0
        v_pin_noisy = v_pin + noise
        
        # Clip to hardware ADC rail [0, Vref]
        v_clipped = np.clip(v_pin_noisy, 0.0, self.vref)
        
        # Quantise to integer digital code
        code = int(np.round((v_clipped / self.vref) * self.max_code))
        code = int(np.clip(code, 0, self.max_code))
        
        # Reconstruct cell voltage using nominal divider ratio
        v_cell_recon = (code / self.max_code) * (self.vref / self.divider_ratio)
        
        return {
            "adc_code": code,
            "v_pin_noisy": float(v_pin_noisy),
            "v_cell_reconstructed": float(v_cell_recon),
            "adc_bits": self.bits,
            "vref": self.vref,
        }
