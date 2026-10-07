# State of Charge (SOC) Digital Twin - PowerShell Launcher
# Virtual ESP32 + TP4056 + 18650 Li-ion Cell
# Amrita School of Engineering - EEE

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "State of Charge (SOC) Digital Twin - Windows PowerShell Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/ and check 'Add Python to PATH'."
    exit 1
}

# 2. Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
    & $activateScript
}

# 3. Dependencies
Write-Host "[INFO] Installing required dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Data Preprocessing
if (-not (Test-Path "data\processed\Discharge_01.parquet")) {
    Write-Host "[INFO] Preprocessing NASA battery cycling data..." -ForegroundColor Yellow
    python -m src.data_prep
}

# 5. Model Training
if (-not (Test-Path "models\mlp_weights.json")) {
    Write-Host "[INFO] Training MLP 3-16-1 neural network..." -ForegroundColor Yellow
    python -m src.train
    python -m src.mlp_infer
}

# 6. Run Pytest Suite
Write-Host "[INFO] Running pytest test suite..." -ForegroundColor Yellow
python -m pytest tests/ -v

# 7. Launch Dashboard
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Launching Live Web Dashboard on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

Start-Process "http://localhost:8000"
python -m src.server
