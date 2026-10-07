"""Unit and Integration Tests for Custom Data Import & ML SOC Prediction."""

import io
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.server import app, sim
from src.custom_import import (
    generate_sample_csv_template,
    parse_and_predict_custom_data,
    export_predictions_csv,
    PREDEFINED_DEFAULT_COLUMNS,
)
from src.twin import DigitalTwin


@pytest.fixture
def client():
    return TestClient(app)


def test_generate_sample_template():
    """Verify sample CSV template generation has predefined columns."""
    csv_text = generate_sample_csv_template(30)
    lines = csv_text.strip().splitlines()
    assert len(lines) == 31 # header + 30 rows
    header = lines[0].split(",")
    for col in PREDEFINED_DEFAULT_COLUMNS:
        assert col in header


def test_parse_default_predefined_columns():
    """Verify parsing data with exact predefined columns (Time, Voltage, Current)."""
    csv_data = """Time,Voltage,Current
0.0,4.15,1.5
5.0,4.10,1.5
10.0,4.05,1.5
15.0,4.00,1.5
20.0,3.95,1.5
"""
    res = parse_and_predict_custom_data(csv_data, filename="test_default.csv")
    assert res["status"] == "success"
    assert res["stats"]["sample_count"] == 5
    assert res["stats"]["v_max"] == 4.15
    assert res["stats"]["v_min"] == 3.95
    assert res["stats"]["i_mean"] == 1.5
    assert len(res["series"]["predicted_soc"]) == 5
    for val in res["series"]["predicted_soc"]:
        assert 0.0 <= val <= 100.0


def test_arbitrary_values_and_pulse_profile():
    """Verify arbitrary values: arbitrary voltages, pulsed currents, charging/negative current."""
    n = 50
    t = np.arange(n) * 2.5 # arbitrary 2.5s sampling rate
    # Arbitrary dynamic profile
    v = np.linspace(4.20, 3.20, n)
    i = np.where(np.sin(np.linspace(0, 10, n)) > 0, 2.0, -1.0) # arbitrary positive & negative current
    
    df = pd.DataFrame({"Time": t, "Voltage": v, "Current": i})
    csv_data = df.to_csv(index=False)
    
    res = parse_and_predict_custom_data(csv_data, filename="arbitrary_pulse.csv")
    assert res["status"] == "success"
    assert res["stats"]["sample_count"] == n
    assert res["stats"]["dt_s"] == 2.5
    # All predictions strictly bounded in [0, 100]
    preds = np.array(res["series"]["predicted_soc"])
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 100.0)


def test_delimiter_sniffing():
    """Verify auto-detection of semicolon and tab delimiters."""
    # Semicolon (NASA format)
    csv_semi = "Time;Voltage;Current;Temperature\n0;4.18;1.0;25.0\n5;4.12;1.0;25.2\n10;4.06;1.0;25.4\n"
    res_semi = parse_and_predict_custom_data(csv_semi)
    assert res_semi["stats"]["sample_count"] == 3
    assert res_semi["stats"]["v_max"] == 4.18

    # Tab
    csv_tab = "Time\tVoltage\tCurrent\n0\t4.15\t2.0\n5\t4.05\t2.0\n"
    res_tab = parse_and_predict_custom_data(csv_tab)
    assert res_tab["stats"]["sample_count"] == 2


def test_column_aliases_and_kelvin_probes():
    """Verify alias mapping: vbat1..4 Kelvin probes, lowercase 'v', 'i', 't'."""
    # NASA Kelvin probes
    csv_nasa = "time;i;temp;vbat1;vbat2;vbat3;vbat4\n0;1.0;24.0;4.10;4.10;4.10;4.10\n5;1.0;24.1;4.08;4.08;4.08;4.08\n"
    res_nasa = parse_and_predict_custom_data(csv_nasa)
    assert res_nasa["stats"]["sample_count"] == 2
    assert abs(res_nasa["stats"]["v_max"] - 4.10) < 1e-3

    # Lowercase aliases
    csv_lower = "t,v,i\n0,4.12,1.2\n5,4.08,1.2\n"
    res_lower = parse_and_predict_custom_data(csv_lower)
    assert res_lower["stats"]["sample_count"] == 2


def test_missing_voltage_error():
    """Verify helpful error when Voltage column is missing."""
    csv_no_v = "Time,Current,Temperature\n0,1.5,25.0\n5,1.5,25.0\n"
    with pytest.raises(ValueError, match="Missing required predefined 'Voltage' column"):
        parse_and_predict_custom_data(csv_no_v)


def test_ground_truth_accuracy_calculation():
    """Verify MAE/RMSE calculation when SOC_True is provided in custom data."""
    csv_data = """Time,Voltage,Current,SOC_True
0.0,4.18,1.5,100.0
5.0,4.14,1.5,95.0
10.0,4.10,1.5,90.0
"""
    res = parse_and_predict_custom_data(csv_data)
    assert res["stats"]["has_ground_truth"] is True
    assert res["stats"]["ml_mae"] is not None
    assert res["stats"]["ml_rmse"] is not None


def test_api_sample_csv_endpoint(client):
    """Test GET /api/sample-csv returns CSV with default columns."""
    res = client.get("/api/sample-csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    text = res.text
    assert "Time" in text
    assert "Voltage" in text
    assert "Current" in text


def test_api_predict_and_export_endpoints(client):
    """Test POST /api/predict-custom and GET /api/export-custom."""
    csv_payload = generate_sample_csv_template(40)
    
    # 1. Predict without twin replay
    res_pred = client.post("/api/predict-custom", json={
        "filename": "my_test_run.csv",
        "content": csv_payload,
        "load_into_twin": False,
    })
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert data["status"] == "success"
    assert data["filename"] == "my_test_run.csv"
    assert data["stats"]["sample_count"] == 40
    assert "predicted_soc" in data["series"]
    assert len(data["preview"]) > 0

    # 2. Export predictions CSV
    res_exp = client.get("/api/export-custom")
    assert res_exp.status_code == 200
    assert "Predicted_SOC_percent" in res_exp.text

    # 3. Load into twin endpoint
    res_twin = client.post("/api/load-custom-twin")
    assert res_twin.status_code == 200
    assert res_twin.json()["status"] == "success"
    assert sim.cycle_key == "custom"

    # 4. Check /api/cycles lists the custom cycle
    res_cycles = client.get("/api/cycles")
    assert res_cycles.status_code == 200
    cycles_data = res_cycles.json()
    assert "custom" in cycles_data
    custom_ids = [c["id"] for c in cycles_data["custom"]]
    assert "custom" in custom_ids

    # 5. Step DigitalTwin on custom data
    frame = sim.twin.step()
    assert "soc_ml" in frame
    assert "chain" in frame
    assert frame["chain"]["cell_v"] > 0


def test_excel_template_and_parsing(tmp_path):
    """Test generating sample Excel file, saving to disk, parsing, and predicting."""
    from src.custom_import import generate_sample_excel_template, export_predictions_excel
    excel_bytes = generate_sample_excel_template(50)
    assert len(excel_bytes) > 1000
    assert excel_bytes.startswith(b"PK\x03\x04")

    excel_file = tmp_path / "test_battery.xlsx"
    excel_file.write_bytes(excel_bytes)

    # 1. Test parsing directly from file path
    res = parse_and_predict_custom_data(excel_file)
    assert res["status"] == "success"
    assert res["stats"]["sample_count"] == 50
    assert len(res["series"]["predicted_soc"]) == 50

    # 2. Test parsing from raw bytes
    res_b = parse_and_predict_custom_data(excel_bytes, filename="test.xlsx")
    assert res_b["status"] == "success"
    assert res_b["stats"]["sample_count"] == 50

    # 3. Test export to Excel
    exp_bytes = export_predictions_excel(res["df_processed"])
    assert len(exp_bytes) > 1000
    assert exp_bytes.startswith(b"PK\x03\x04")
    df_readback = pd.read_excel(io.BytesIO(exp_bytes))
    assert "Predicted_SOC_percent" in df_readback.columns


def test_api_sample_xlsx_and_export_excel(client):
    """Test GET /api/sample-xlsx and GET /api/export-custom-excel."""
    import base64
    from src.custom_import import generate_sample_excel_template

    # 1. Download Excel template
    res_tpl = client.get("/api/sample-xlsx")
    assert res_tpl.status_code == 200
    assert "spreadsheetml" in res_tpl.headers["content-type"]
    assert len(res_tpl.content) > 1000

    # 2. Upload Excel as base64 data URL
    b64_content = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + base64.b64encode(res_tpl.content).decode("ascii")
    res_pred = client.post("/api/predict-custom", json={
        "filename": "my_uploaded_excel.xlsx",
        "content": b64_content,
        "load_into_twin": False,
    })
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert pred_data["status"] == "success"
    assert pred_data["filename"] == "my_uploaded_excel.xlsx"

    # 3. Export predictions to Excel
    res_exp = client.get("/api/export-custom-excel")
    assert res_exp.status_code == 200
    assert "spreadsheetml" in res_exp.headers["content-type"]
    assert len(res_exp.content) > 1000

