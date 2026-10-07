"""Cell Twin Component.

Replays preprocessed NASA battery cycles sample-by-sample (5s dt).
Supports play, pause, seek, loop, and step.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class CellTwin:
    """Simulates 18650 Li-ion cell replay from preprocessed NASA cycling data."""
    
    def __init__(self, cycle_df: pd.DataFrame, cycle_name: str = "custom"):
        self.df = cycle_df.reset_index(drop=True)
        self.cycle_name = cycle_name
        self.total_samples = len(self.df)
        self.cursor = 0
        self.is_done = False
        
    def reset(self) -> None:
        """Reset replay cursor to t=0."""
        self.cursor = 0
        self.is_done = False
        
    def seek(self, sample_idx: int) -> None:
        """Seek cursor to specific sample index."""
        self.cursor = int(np.clip(sample_idx, 0, self.total_samples - 1))
        self.is_done = (self.cursor >= self.total_samples - 1)
        
    def step(self) -> Dict[str, Any]:
        """Fetch next sample and advance internal cursor."""
        if self.cursor >= self.total_samples:
            self.is_done = True
            row = self.df.iloc[-1]
        else:
            row = self.df.iloc[self.cursor]
            self.cursor += 1
            if self.cursor >= self.total_samples:
                self.is_done = True
                
        return {
            "index": int(row.name if hasattr(row, "name") else self.cursor - 1),
            "time_s": float(row.get("time_s", (self.cursor - 1) * 5.0)),
            "v_true": float(row.get("V", 3.7)),
            "i_true": float(row.get("I", 0.0)),
            "soc_true": float(row.get("soc_true", row.get("soc_ml", 100.0))),
            "temperature_c": float(row.get("temperature_c", 25.0)),
            "is_done": self.is_done,
        }
