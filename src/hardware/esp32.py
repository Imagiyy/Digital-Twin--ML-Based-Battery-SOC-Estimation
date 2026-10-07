"""ESP32 Firmware Logic Component.

Implements the embedded microcontroller runtime:
- Maintains a 12-sample ring buffer for causal 60s moving average
- Uses expanding window warm-up identical to training
- Runs scalar MLP 3-16-1 forward pass
- Streams JSON Wi-Fi telemetry payloads:
  {"v": 3.915, "i": -1.02, "soc": 58.7, "status": "CHARGING"}
"""

from collections import deque
import json
from typing import Dict, Any, List, Optional
import numpy as np

from src.mlp_infer import PureNumpyMLP


class Esp32Firmware:
    """Simulates on-chip ESP32 C++ firmware execution loop."""
    
    def __init__(self, model: PureNumpyMLP, window_size: int = 12):
        self.model = model
        self.window_size = window_size
        self.ring_buffer = deque(maxlen=window_size)
        self.sample_count = 0
        
    def reset(self) -> None:
        """Reset internal firmware ring buffer and sample counter."""
        self.ring_buffer.clear()
        self.sample_count = 0
        
    def step(
        self,
        v_recon: float,
        i_measured: float,
        charger_status: str,
    ) -> Dict[str, Any]:
        """Execute one 5s firmware cycle."""
        # 1. Update circular buffer
        self.ring_buffer.append(v_recon)
        self.sample_count += 1
        
        # 2. Causal 60s moving average (expanding window warm-up)
        # Identical to training compute_causal_moving_average
        v_smooth = sum(self.ring_buffer) / len(self.ring_buffer)
        
        # 3. On-chip MLP inference (scalar pure-python forward pass)
        import time
        t_start = time.perf_counter_ns()
        soc_estimate = self.model.predict_scalar(v_recon, i_measured, v_smooth)
        t_us = max(1.0, (time.perf_counter_ns() - t_start) / 1000.0)
        
        # 4. Standard Wi-Fi telemetry payload (per Master Prompt specification)
        wifi_payload = {
            "v": round(v_recon, 3),
            "i": round(i_measured, 2),
            "soc": round(soc_estimate, 1),
            "status": charger_status,
        }
        
        return {
            "v_smooth": float(v_smooth),
            "soc_estimate": float(soc_estimate),
            "infer_time_us": float(round(t_us, 2)),
            "wifi_payload": wifi_payload,
            "wifi_payload_json": json.dumps(wifi_payload),
        }
