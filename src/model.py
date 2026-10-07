"""MLP 3-16-1 Architecture Definition, Weight Exporter, and C Header Generator.

Features: [V, I, V_smooth]
Hidden layer: 16 units, tanh activation
Output layer: 1 unit, linear activation, clipped to [0, 100]
Total trainable parameters: 81 (3*16 + 16 + 16*1 + 1)
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np


class MLP3_16_1:
    """Canonical MLP 3-16-1 State of Charge estimator."""
    
    def __init__(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: float,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
    ):
        """Initialize with weights and scaler parameters.
        
        W1: shape (16, 3) or (3, 16)
        b1: shape (16,)
        W2: shape (16,) or (1, 16)
        b2: float scalar
        """
        # Store in standard shape: W1 (16, 3), W2 (16,)
        self.W1 = np.asarray(W1, dtype=np.float32)
        if self.W1.shape == (3, 16):
            self.W1 = self.W1.T
        assert self.W1.shape == (16, 3), f"Expected W1 shape (16, 3), got {self.W1.shape}"
        
        self.b1 = np.asarray(b1, dtype=np.float32).reshape(16)
        
        self.W2 = np.asarray(W2, dtype=np.float32).reshape(16)
        self.b2 = float(b2)
        
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32).reshape(3)
        self.scaler_std = np.asarray(scaler_std, dtype=np.float32).reshape(3)
        
    @property
    def total_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return (16 * 3) + 16 + 16 + 1 # 48 + 16 + 16 + 1 = 81
        
    def predict_sample(self, v: float, i: float, v_smooth: float) -> float:
        """Scalar pure-Python inference for real-time per-sample embedded simulation."""
        # 1. Scale inputs
        x0 = (v - self.scaler_mean[0]) / self.scaler_std[0]
        x1 = (i - self.scaler_mean[1]) / self.scaler_std[1]
        x2 = (v_smooth - self.scaler_mean[2]) / self.scaler_std[2]
        
        # 2. Hidden layer with tanh activation (16 units)
        soc_acc = self.b2
        for j in range(16):
            z = self.b1[j] + self.W1[j, 0] * x0 + self.W1[j, 1] * x1 + self.W1[j, 2] * x2
            # tanh(z)
            a = np.tanh(z)
            soc_acc += self.W2[j] * a
            
        # 3. Clip output to [0, 100]
        if soc_acc < 0.0:
            return 0.0
        elif soc_acc > 100.0:
            return 100.0
        return float(soc_acc)
        
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized NumPy inference for batch evaluation."""
        X_arr = np.asarray(X, dtype=np.float32)
        # Standardize
        X_norm = (X_arr - self.scaler_mean) / self.scaler_std
        # Hidden: (N, 3) @ (3, 16) + (16,)
        Z = np.dot(X_norm, self.W1.T) + self.b1
        A = np.tanh(Z)
        # Output: (N, 16) @ (16,) + scalar
        Y = np.dot(A, self.W2) + self.b2
        return np.clip(Y, 0.0, 100.0)
        
    def export_weights_json(self, export_path: Path, metadata: Dict[str, Any] = None) -> None:
        """Export model parameters, scaler constants, and architecture to JSON."""
        export_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_architecture": "MLP 3-16-1",
            "total_parameters": self.total_parameters,
            "features": ["V", "I", "V_smooth"],
            "activation_hidden": "tanh",
            "activation_output": "linear_clipped",
            "clip_range": [0.0, 100.0],
            "scaler": {
                "mean": self.scaler_mean.tolist(),
                "std": self.scaler_std.tolist(),
            },
            "weights": {
                "W1": self.W1.tolist(),
                "b1": self.b1.tolist(),
                "W2": self.W2.tolist(),
                "b2": self.b2,
            },
            "metadata": metadata or {},
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Model] Exported weights JSON to {export_path}")
        
    def export_weights_c_header(self, header_path: Path) -> None:
        """Generate firmware C header mlp_weights.h for ESP32 compilation."""
        header_path.parent.mkdir(parents=True, exist_ok=True)
        with open(header_path, "w", encoding="utf-8") as f:
            f.write("/*\n")
            f.write(" * Auto-generated MLP 3-16-1 Weights Header for ESP32 Battery Twin\n")
            f.write(" * Parameters: 81 (float32)\n")
            f.write(" * Inputs: [V (V), I (A), V_smooth (V)]\n")
            f.write(" */\n\n")
            f.write("#ifndef MLP_WEIGHTS_H\n")
            f.write("#define MLP_WEIGHTS_H\n\n")
            f.write("#define MLP_INPUT_DIM 3\n")
            f.write("#define MLP_HIDDEN_DIM 16\n")
            f.write("#define MLP_TOTAL_PARAMS 81\n\n")
            
            # Scaler mean & std
            f.write("// Feature standardisation constants (fit on training set only)\n")
            f.write(f"static const float SCALER_MEAN[3] = {{{self.scaler_mean[0]:.7f}f, {self.scaler_mean[1]:.7f}f, {self.scaler_mean[2]:.7f}f}};\n")
            f.write(f"static const float SCALER_STD[3]  = {{{self.scaler_std[0]:.7f}f, {self.scaler_std[1]:.7f}f, {self.scaler_std[2]:.7f}f}};\n\n")
            
            # W1 (16 x 3)
            f.write("// Hidden layer weights W1 (16 neurons x 3 inputs)\n")
            f.write("static const float W1[16][3] = {\n")
            for j in range(16):
                f.write(f"    {{{self.W1[j,0]:.7f}f, {self.W1[j,1]:.7f}f, {self.W1[j,2]:.7f}f}},\n")
            f.write("};\n\n")
            
            # b1 (16)
            f.write("// Hidden layer bias b1 (16 neurons)\n")
            f.write("static const float B1[16] = {\n    ")
            for j in range(16):
                f.write(f"{self.b1[j]:.7f}f, ")
            f.write("\n};\n\n")
            
            # W2 (16)
            f.write("// Output layer weights W2 (16 weights)\n")
            f.write("static const float W2[16] = {\n    ")
            for j in range(16):
                f.write(f"{self.W2[j]:.7f}f, ")
            f.write("\n};\n\n")
            
            # b2 scalar
            f.write(f"// Output layer bias b2\n")
            f.write(f"static const float B2 = {self.b2:.7f}f;\n\n")
            f.write("#endif // MLP_WEIGHTS_H\n")
        print(f"[Model] Exported C header to {header_path}")
