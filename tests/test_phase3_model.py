"""Unit tests for Phase 3: Model Architecture, Inference Parity, and Golden Vectors."""

import json
from pathlib import Path
import numpy as np
import pytest
import joblib

from src.config import get_path
from src.model import MLP3_16_1
from src.mlp_infer import PureNumpyMLP


@pytest.fixture
def trained_artifacts():
    weights_path = get_path("models_dir") / "mlp_weights.json"
    joblib_path = get_path("models_dir") / "mlp_model.joblib"
    assert weights_path.exists(), "mlp_weights.json missing"
    assert joblib_path.exists(), "mlp_model.joblib missing"
    
    model_np = PureNumpyMLP(weights_path)
    model_sklearn = joblib.load(joblib_path)
    return model_np, model_sklearn


def test_parameter_count(trained_artifacts):
    """Verify exact count of 81 parameters."""
    model_np, _ = trained_artifacts
    total = (16 * 3) + 16 + (16 * 1) + 1
    assert model_np.W1.size + model_np.b1.size + model_np.W2.size + 1 == total
    assert total == 81


def test_numpy_vs_scalar_python_parity(trained_artifacts):
    """Verify scalar python inference exactly matches vectorized numpy inference."""
    model_np, _ = trained_artifacts
    rng = np.random.default_rng(42)
    v_test = rng.uniform(3.0, 4.2, size=1000)
    i_test = rng.uniform(-3.0, 3.0, size=1000)
    vs_test = v_test + rng.normal(0, 0.02, size=1000)
    
    X = np.column_stack([v_test, i_test, vs_test])
    batch_preds = model_np.predict_batch(X)
    
    scalar_preds = np.array([
        model_np.predict_scalar(float(v_test[k]), float(i_test[k]), float(vs_test[k]))
        for k in range(1000)
    ])
    
    np.testing.assert_allclose(batch_preds, scalar_preds, atol=1e-5)


def test_numpy_vs_sklearn_framework_parity(trained_artifacts):
    """Verify pure-NumPy inference reproduces scikit-learn framework output to within 1e-5."""
    model_np, model_sklearn = trained_artifacts
    rng = np.random.default_rng(100)
    n_samples = 10000
    v_test = rng.uniform(3.0, 4.2, size=n_samples)
    i_test = rng.uniform(-3.0, 3.0, size=n_samples)
    vs_test = v_test + rng.normal(0, 0.02, size=n_samples)
    
    X = np.column_stack([v_test, i_test, vs_test])
    
    # NumPy inference
    np_preds = model_np.predict_batch(X)
    
    # Sklearn inference
    X_norm = (X - model_np.mean) / model_np.std
    sk_preds = np.clip(model_sklearn.predict(X_norm), 0.0, 100.0)
    
    # Maximum absolute difference
    max_diff = np.max(np.abs(np_preds - sk_preds))
    assert max_diff < 1e-4, f"Max difference {max_diff} exceeded tolerance 1e-4"


def test_golden_test_vectors(trained_artifacts):
    """Verify golden vectors match model predictions."""
    model_np, _ = trained_artifacts
    golden_file = Path(__file__).resolve().parent / "golden_vectors.json"
    assert golden_file.exists(), "golden_vectors.json missing"
    
    with open(golden_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for vec in data["vectors"]:
        v = vec["v"]
        i = vec["i"]
        vs = vec["v_smooth"]
        exp = vec["expected_soc"]
        act = model_np.predict_scalar(v, i, vs)
        assert abs(act - exp) < 1e-4, f"Golden vector {vec['index']} mismatch: expected {exp}, got {act}"
