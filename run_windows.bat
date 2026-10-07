@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo State of Charge (SOC) Digital Twin - Windows Launcher
echo Virtual ESP32 + TP4056 + 18650 Li-ion Cell
echo Amrita School of Engineering - EEE
echo ======================================================================

:: 1. Verify Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not found in system PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/
    echo Ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

:: 2. Create and activate virtual environment (if not present)
if not exist ".venv" (
    echo [INFO] Creating virtual environment (.venv)...
    python -m venv .venv
)

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

:: 3. Install dependencies
echo [INFO] Installing required dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Preprocess data (if processed files missing)
if not exist "data\processed\Discharge_01.parquet" (
    echo [INFO] Running Phase 2: Data preprocessing and causal feature engineering...
    python -m src.data_prep
)

:: 5. Train model (if weights missing)
if not exist "models\mlp_weights.json" (
    echo [INFO] Running Phase 3: Model training with noise augmentation...
    python -m src.train
    python -m src.mlp_infer
)

:: 6. Run automated test suite
echo [INFO] Running test verification suite...
python -m pytest tests\ -v

:: 7. Launch web dashboard
echo ======================================================================
echo Starting Live Web Dashboard on http://localhost:8000
echo Opening web browser...
echo (Press Ctrl+C to stop the server)
echo ======================================================================
start http://localhost:8000
python -m src.server

pause
