"""Integration tests for Phase 5: FastAPI Backend REST Endpoints and WebSocket Stream."""

import json
import pytest
from fastapi.testclient import TestClient

from src.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify health check endpoint returns 200 and valid JSON."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_cycles_endpoint(client):
    """Verify cycles list returns held-out and training cycles."""
    res = client.get("/api/cycles")
    assert res.status_code == 200
    data = res.json()
    assert "held_out" in data
    assert len(data["held_out"]) >= 4
    held_ids = [c["id"] for c in data["held_out"]]
    assert "pair10" in held_ids
    assert "charge10" in held_ids


def test_control_and_params_endpoints(client):
    """Verify control and parameter modification endpoints."""
    # Control: pause
    res_ctrl = client.post("/api/control", json={"action": "pause"})
    assert res_ctrl.status_code == 200
    assert res_ctrl.json()["is_playing"] is False
    
    # Params: update noise and offset
    res_param = client.post("/api/params", json={"noise_mv": 8.0, "offset_ma": 40.0})
    assert res_param.status_code == 200
    assert res_param.json()["noise_mv"] == 8.0
    assert res_param.json()["offset_ma"] == 40.0


def test_model_and_results_endpoints(client):
    """Verify model footprint and precomputed results endpoints."""
    res_model = client.get("/api/model")
    assert res_model.status_code == 200
    model_data = res_model.json()
    assert model_data["total_parameters"] == 81
    assert "footprint" in model_data
    
    res_res = client.get("/api/results")
    assert res_res.status_code == 200
    results_data = res_res.json()
    assert "held_out_results" in results_data
    assert len(results_data["held_out_results"]) == 4


def test_websocket_stream_frame_schema(client):
    """Verify telemetry frame and Wi-Fi payload schema."""
    from src.server import sim
    frame = sim.twin.step()
    
    # Check mandatory telemetry keys
    assert "soc_ml" in frame
    assert "soc_true" in frame
    assert "chain" in frame
    assert "wifi_payload" in frame
    assert "status" in frame
    
    payload = frame["wifi_payload"]
    assert "v" in payload and "i" in payload and "soc" in payload and "status" in payload
    
    # Check running metrics include RMSE and inference timing
    assert "running_ml_rmse" in frame["metrics"]
    assert "running_ml_mae" in frame["metrics"]
    assert "infer_time_us" in frame["metrics"]


def test_static_and_report_assets_served(client):
    """Verify web static assets and report figures are properly served."""
    # Index
    res_index = client.get("/")
    assert res_index.status_code == 200
    
    # Static CSS & JS
    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    
    # Report figures
    res_fig = client.get("/report/figures/held_out_tracking.png")
    assert res_fig.status_code == 200
    assert "image" in res_fig.headers.get("content-type", "")
