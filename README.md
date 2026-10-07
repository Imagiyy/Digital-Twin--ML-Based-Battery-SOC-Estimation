# Machine-Learning SOC Estimation Digital Twin
### Virtual ESP32 + TP4056 + 18650 Li-ion Cell (Driven by NASA Cycling Data)

**B.Tech Engineering Project** &middot; Electrical & Electronics Engineering (EEE) &middot; Amrita School of Engineering, Amrita Vishwa Vidyapeetham

---

## Quick Start

### Option A: Linux / macOS (Using Makefile)

Run the entire system offline from a clean clone with four simple commands:

```bash
make setup       # 1. Verify dependencies
make prep        # 2. Resample 60 NASA files to 5s grid & compute causal features
make train       # 3. Train MLP 3-16-1 with noise augmentation & export C header
make serve       # 4. Launch live Web Dashboard on http://localhost:8000
```

To run all automated verification suites (Python pytest + Host-side C tests):
```bash
make test        # Runs all 39 unit & integration tests (prep, model, hardware, server, import, SOH, Kalman)
make test-c      # Compiles & verifies host C parity against golden vectors
```

---

### Option B: Windows (Command Prompt, PowerShell, or 1-Click Batch)

#### 1. One-Click Automated Run (Fastest)
Double-click `run_windows.bat` in File Explorer, or run in Command Prompt:
```cmd
run_windows.bat
```
*(Or in PowerShell: `.\run_windows.ps1`)*

This script automatically creates a `.venv` virtual environment, installs dependencies, preprocesses data, trains the MLP model, runs all 39 pytest verification tests, and opens `http://localhost:8000` in your default browser.

#### 2. Manual Step-by-Step (Command Prompt `cmd.exe`)

Open Command Prompt in the project folder and run:

```cmd
:: 1. Create and activate a virtual environment
python -m venv .venv
call .venv\Scripts\activate.bat

:: 2. Install requirements
pip install -r requirements.txt

:: 3. Preprocess 60 NASA battery cycling files (5s grid & causal features)
python -m src.data_prep

:: 4. Train MLP 3-16-1 neural network & export weights JSON and C header
python -m src.train
python -m src.mlp_infer

:: 5. (Optional) Run all held-out evaluations and experiments
python -m src.evaluate
python -m src.experiments.baselines
python -m src.experiments.ablations
python -m src.experiments.robustness
python -m src.experiments.drift
python -m src.experiments.explain

:: 6. Launch Live Web Dashboard
python -m src.server
```
Then open **`http://localhost:8000`** in Chrome, Edge, or Firefox.

#### 3. Running Automated Tests on Windows

```cmd
:: Run all 39 Python unit and integration tests
python -m pytest tests/ -v
```

#### 4. Running Host-Side C Parity Test on Windows (Optional)

If you have MinGW `gcc` (e.g. from MSYS2, Git for Windows SDK, or w64devkit):
```cmd
gcc -O3 -I firmware\include firmware\test\test_c_parity.c -o firmware\test\test_c_parity.exe
firmware\test\test_c_parity.exe
```

Or using Microsoft Visual C++ (`cl` from Visual Studio Developer Command Prompt):
```cmd
cl /O2 /I firmware\include firmware\test\test_c_parity.c /Fe:firmware\test\test_c_parity.exe
firmware\test\test_c_parity.exe
```

*(Note: If you have Make installed via Chocolatey `choco install make` or Winget `winget install ezwinports.make`, you can also use `make prep`, `make train`, `make serve`, `make test` directly on Windows).*

#### 5. Testing Using Excel (.xlsx / .xls) Files

You can test custom battery cycle data using Excel spreadsheets through either the **Python CLI Tool** or the **Live Web Dashboard**:

##### A. Command Line Interface (CLI)

A pre-packaged Excel test file [`sample_battery_data.xlsx`](file:///home/abrar/Downloads/BMS/proj/sample_battery_data.xlsx) is included in the project root:

```bash
# 1. Run inference directly on an Excel file
python src/predict_excel.py sample_battery_data.xlsx

# 2. Run inference, export predictions to Excel, and generate an evaluation chart
python src/predict_excel.py sample_battery_data.xlsx --export predictions.xlsx --plot report/figures/excel_test_result.png

# 3. Generate a fresh Excel template with predefined columns
python src/predict_excel.py --generate-sample my_test_template.xlsx --samples 150
```

##### B. Web Dashboard (Drag-and-Drop)

1. Run `make serve` or `python -m src.server` and open **`http://localhost:8000`**.
2. Navigate to the **"Custom Data Import"** tab.
3. Click or drag-and-drop your `.xlsx` file into the upload zone.
4. Click **"🚀 Predict SOC"**:
   - The neural network computes real-time predictions.
   - Shows summary KPIs, voltage/current charts, and MAE/RMSE scorecard (if `SOC_True` column is present).
   - Click **"📊 Export Excel (.xlsx)"** to download the annotated spreadsheet.
   - Check **"Load into Live Digital Twin replay"** to stream your Excel cycle live into the telemetry dashboard.

##### Expected Excel Columns:
- **`Time`** (or `Time_s`, `t`): Timestamp in seconds.
- **`Voltage`** (or `V`, `Vbat`): Cell terminal voltage in Volts (e.g. 3.0 V – 4.2 V).
- **`Current`** (or `I`, `Amps`): Load current in Amperes (positive for discharge, negative for charge).
- **`Temperature`** *(Optional)*: Cell temperature in °C (defaults to 25.0 °C if omitted).
- **`SOC_True`** *(Optional)*: Reference actual SOC % (if provided, computes validation MAE & RMSE).

---

## 1. Project Overview & Physical Objective

State of Charge (SOC) of a lithium-ion battery cannot be measured directly. Traditional Coulomb Counting integrates current over time:

$$\text{SOC}(t) = \text{SOC}(0) - \frac{1}{C_{\text{nominal}}} \int_0^t i(\tau) \, d\tau$$

However, Coulomb Counting suffers from:
1. **Sensor Bias Accumulation**: A DC offset (e.g. 30 mA) steadily drifts integrated capacity over multi-hour runs.
2. **Initial SOC Sensitivity**: An inaccurate $\text{SOC}(0)$ causes permanent offset error that never self-corrects.
3. **OCV Rest Requirement**: Open-circuit voltage lookup requires long relaxation periods (hours) unsuitable under active load due to internal resistance ($IR$) drop: $V_{\text{terminal}} = V_{\text{OCV}} - I \cdot R_0$.

This project implements an end-to-end **Digital Twin** modeling an embedded ESP32 monitoring system. A compact neural network (**MLP 3-16-1**, $\tanh$ hidden layer, 81 parameters) estimates real-time SOC directly from terminal voltage, current, and a causal 60 s moving average voltage.

### Virtual Hardware Signal Flow

$$\text{NASA 18650 Cell Data} \longrightarrow \text{Divider } (\div 2) \longrightarrow \text{ESP32 12-bit ADC} \longrightarrow \text{Virtual Firmware} \longrightarrow \text{Wi-Fi Telemetry JSON} \longrightarrow \text{Web UI}$$

The virtual hardware chain models:
- **Resistor Voltage Divider**: 0.5 ratio step-down to fit the ESP32 ADC rail (0 - 3.3 V).
- **ESP32 12-bit SAR ADC**: Codes $0 - 4095$, $V_{\text{ref}} = 3.3\text{ V}$, plus $5\text{ mV}$ RMS Gaussian thermal noise.
- **Current Shunt Sensor**: Injected with $+30\text{ mA}$ DC bias offset.
- **TP4056 Linear Charger Controller**: State machine handling `CHARGING`, `DISCHARGING`, and `CHARGED_STANDBY` CV termination with `CHRG` (Red) and `STDBY` (Green) LED telemetry.

---

## 2. Quantitative Acceptance Results

Evaluated on the **fixed 7 held-out test cycles** (Discharge 02, Pair 10, Pair 20, Pair 30) with **$5\text{ mV}$ ADC noise** and **$30\text{ mA}$ current sensor offset** active:

| Held-Out Test Cycle | Rate (A) | Duration (h) | ML MAE (%) | ML RMSE (%) | ML Max (%) | Coulomb Counting MAE (%) | Target (< 2-3%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Discharge 02** | 0.5 A | 5.87 h | **1.28 %** | 1.69 % | 5.46 % | 3.32 % | **MET ✓** |
| **Pair 10 (Dis + Chg)** | 1.0 A | 6.06 h | **1.16 %** | 1.57 % | 6.19 % | 2.37 % | **MET ✓** |
| **Pair 20 (Dis + Chg)** | 2.0 A | 3.08 h | **1.07 %** | 1.49 % | 11.05 % | 0.95 % | **MET ✓** |
| **Pair 30 (Dis + Chg)** | 3.0 A | 2.18 h | **1.49 %** | 2.12 % | 11.29 % | 2.33 % | **MET ✓** |

> **Key Takeaway**: Every single held-out test cycle achieves an ML Mean Absolute Error under 1.5 percentage points, exceeding the project acceptance threshold (< 2-3%). On long runs (Discharge 02), the ML estimator beats Coulomb Counting (1.28% vs 3.32%) as current sensor offset accumulates.

---

## 3. State of Health (SOH) Estimation: Dual Aging Algorithms

Battery health degrades over cycling due to two distinct physical aging mechanisms: loss of active lithium inventory (capacity fade) and solid electrolyte interphase (SEI) thickening / contact resistance increase (resistance fade). This project implements both estimators and fuses them:

### 1. Capacity Fade Algorithm ($SOH_C$)
Measures the degradation of usable cell charge capacity relative to fresh nominal capacity ($C_{\text{nominal}} = 2.65\text{ Ah}$):
$$SOH_C = \frac{C_{\text{actual}}}{C_{\text{nominal}}} \times 100\%$$
- **Real-Time Ah-Throughput Degradation Model**: Integrates cumulative charge/discharge throughput $Ah_{\text{throughput}} = \frac{1}{3600}\int |i(\tau)| d\tau$ using a semi-empirical Fickian square-root diffusion power-law:
  $$\Delta C = \alpha_{\text{fade}} \cdot \sqrt{Ah_{\text{throughput}}}$$
- **Cycle-by-Cycle NASA Integration**: Computes real measured capacity across all 30 NASA discharge cycles, tracking capacity loss from $2.65\text{ Ah}$ down to $2.29\text{ Ah}$.

### 2. Resistance Fade Algorithm ($SOH_R$)
Measures the growth of internal ohmic series resistance ($R_0$) relative to fresh state ($R_{\text{fresh}} = 55\text{ m}\Omega$) and end-of-life threshold ($R_{\text{eol}} = 110\text{ m}\Omega$, $+100\%$ resistance growth):
$$SOH_R = \max\left(0, \min\left(100, \frac{R_{\text{eol}} - R_0}{R_{\text{eol}} - R_{\text{fresh}}} \times 100\%\right)\right)$$
- **Dynamic Load-Step Ohmic Resistance Extraction**: Detects instantaneous current steps ($|\Delta I| \ge 150\text{ mA}$) and voltage drops:
  $$R_{0,\text{sample}} = \frac{|\Delta V_{\text{terminal}}|}{|\Delta I_{\text{cell}}|}$$
- **Recursive Exponential Filter**: Smooths sampled resistance measurements:
  $$R_{0,k} = (1 - \gamma) R_{0,k-1} + \gamma R_{0,\text{sample}} \quad (\gamma = 0.015)$$

### 3. Combined Dual-Metric SOH & Remaining Useful Life (RUL)
Fuses capacity loss and impedance rise into a single robust health score:
$$SOH = w_c \cdot SOH_C + w_r \cdot SOH_R \quad (w_c = 0.60, w_r = 0.40)$$
- **Health Classification**:
  - `EXCELLENT` ($SOH \ge 92\%$): Fresh cell, minimal SEI degradation.
  - `GOOD` ($85\% \le SOH < 92\%$): Normal operational aging.
  - `MODERATE AGING` ($80\% \le SOH < 85\%$): Approaching replacement threshold.
  - `END OF LIFE` ($SOH < 80\%$): Battery retired per automotive/industrial standard.
- **RUL Projection**: Calculates remaining equivalent full cycles to the 80% EOL boundary based on degradation rate.

---

## 4. Nonlinear Kalman Filters for SOC (EKF & SPKF)

To benchmark against classical physics-based state estimators, the platform includes a **1-RC Thevenin Equivalent Circuit Model (ECM)**:
- States: $x = [\text{SOC}, V_p]^T$ (SOC normalized in $[0, 1]$, polarization voltage across RC branch in Volts).
- State Transition: $\text{SOC}_k = \text{SOC}_{k-1} - \frac{\eta \Delta t}{Q_n} I_{k-1}$, $V_{p,k} = V_{p,k-1} e^{-\Delta t/\tau} + R_1 (1 - e^{-\Delta t/\tau}) I_{k-1}$ ($\tau = R_1 C_1 = 60\text{ s}$).
- Output Equation: $V_{\text{terminal}} = V_{\text{OCV}}(\text{SOC}) - V_p - I \cdot R_0$.
- OCV Model: Analytical 6th-order polynomial fit to the 18650 cell characteristic curve:
  $$V_{\text{OCV}}(s) = -38.15 s^6 + 132.89 s^5 - 181.70 s^4 + 123.61 s^3 - 43.20 s^2 + 7.72 s + 3.00$$

### 1. Extended Kalman Filter (EKF)
- Linearizes the non-linear observation function using the analytical Jacobian:
  $$H_k = \left[ \frac{\partial V_{\text{terminal}}}{\partial \text{SOC}}, \, \frac{\partial V_{\text{terminal}}}{\partial V_p} \right] = \left[ \frac{dV_{\text{OCV}}}{d\text{SOC}}, \, -1 \right]$$
- Computes Kalman gain $K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1}$ and updates error covariance.

### 2. Sigma-Point Kalman Filter (SPKF / UKF)
- Eliminates Jacobian linearization errors by generating $2L+1 = 5$ deterministic sigma points using the scaled unscented transform ($\alpha = 1.0, \beta = 2.0, \kappa = 1.0$).
- Propagates sigma points directly through the non-linear $V_{\text{OCV}}(s)$ equation, accurately capturing mean and covariance around the steep knee regions of the discharge curve.

---

## 5. Multi-Algorithm Comparison Scorecard & Best Algorithm Recommendation

Evaluated across real NASA 18650 cycling files under standard hardware noise ($5\text{ mV}$ ADC noise $+ 30\text{ mA}$ current sensor DC offset):

| Rank | Algorithm | Category | Avg MAE (%) | MCU Latency (&mu;s) | Flash/RAM (Bytes) | Sensor Bias Resilience | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---|:---|
| 🥇 **1** | **MLP 3-16-1 (Proposed)** | Neural Network | **1.25 %** | **6.03 &mu;s** | **348 B** | **EXCELLENT (Bounded < 1.5%)** | **RECOMMENDED** |
| 🥈 **2** | **Random Forest** | Tree Ensemble | 2.04 % | 24.8 &mu;s | >120,000 B | High (Orthogonal splits) | Heavy ROM Overhead |
| 🥉 **3** | **Linear Regression** | Linear Statistical | 3.02 % | 2.40 &mu;s | 32 B | Low (Linear limitation) | Underfits Non-linear Knee |
| 4 | **SPKF (Sigma-Point Kalman)** | Nonlinear Physics Filter | 6.02 % | 118.5 &mu;s | 3,400 B | Moderate (Drifts with DC Bias) | Best Physics-Based |
| 5 | **EKF (Extended Kalman)** | Linearized Physics Filter | 6.02 % | 42.1 &mu;s | 1,850 B | Moderate (Drifts with DC Bias) | Classical Baseline |
| 6 | **Coulomb Counting** | Current Integration | 2.24 %* | 1.10 &mu;s | 16 B | POOR (>13% drift on long runs) | Unreliable in Field |
| 7 | **OCV Lookup Table** | Static Lookup | 9.75 % | 4.50 &mu;s | 120 B | Fails under active load ($IR$ drop) | Rest Only |

*\*Note: Coulomb Counting average MAE shown with perfect initial SOC. When initialized with -10% error, Coulomb Counting error explodes to 11.23% MAE.*

### Why MLP 3-16-1 is the Best Algorithm for Embedded BMS (ESP32)
1. **Lowest Estimation Error (1.25% MAE)**: Consistently outperforms physics filters across all C-rates (0.5A to 3.0A).
2. **Comparison with ML Baselines (Random Forest vs Linear Regression)**:
   - **Linear Regression (3.02% MAE)**: Lacks capacity to model the flat central plateau and non-linear exponential knees of Li-ion open-circuit voltage curves.
   - **Random Forest (2.04% MAE)**: Effectively partitions non-linear feature spaces, but the 30-tree ensemble requires >120 KB flash memory and produces non-continuous step estimates, making it impractical for resource-constrained microcontrollers.
   - **Proposed MLP 3-16-1 (1.25% MAE)**: Tanh hidden units generate smooth continuous predictions, achieving the lowest error while requiring only 81 parameters (348 bytes) and 6.03 &mu;s latency.
3. **Sensor DC Bias Immunity**: Under $+30\text{ mA}$ current sensor offset over multi-hour runs, standard EKF and SPKF state transitions integrate the bias, causing estimated SOC to drift. The MLP's causal input vector ($[V, I, V_{\text{smooth}}]$) anchors the estimate to terminal voltage dynamics, keeping error strictly bounded.
4. **Ultra-Low Embedded Footprint**: Executes in **6.03 &mu;s** per sample with only **348 bytes** of storage and zero matrix math, leaving 99.9% of the ESP32 CPU free for Wi-Fi and safety monitoring.
5. **Zero Laboratory Parameter Tuning**: EKF/SPKF require extensive climatic chamber testing to calibrate $R_0, R_1, C_1$ across temperatures and SOC levels, whereas the MLP trains end-to-end directly from standard cycling logs.

---

### Battery Estimation Methodologies: Modern AI & Hybrid Frameworks

#### 4. Machine Learning / AI Approaches
- **LSTM (Long Short-Term Memory)**: Learns degradation patterns from dynamic time-series sequences.
- **Random Forest / Gradient Boosting**: Non-linear mapping of extracted features &rarr; SOH (and SOC tabular baselines).
- **Neural Networks (MLP/CNN)**: End-to-end learning directly on voltage, current, temperature, and cycle count.
- **Pros**: No explicit physical/electrochemical model needed; naturally captures complex multi-physics degradation.
- **Cons**: Requires large labeled training datasets; higher computational and memory overhead for SOH on edge microcontrollers.

#### 5. Hybrid Methods (Industry Standard)
Combines two or more complementary physics-based and data-driven methods:
- **Capacity + EIS**: Periodic electrochemical impedance spectroscopy paired with continuous Ah capacity tracking.
- **Model-Based + ML**: Physical Kalman filter (EKF/SPKF) primary state observer + Neural Network correction layer for residual error.
- **Incremental Capacity Analysis (ICA) + ML**: Differential capacity ($dQ/dV$) peak feature extraction paired with ML regression for SOH estimation.
- **Pros**: Combines thermodynamic physical bounds with data-driven non-linear residual correction.
- **Cons**: Requires complex high-frequency instrumentation (AC perturbation) and multi-parameter co-calibration.

---

## 6. Windows Alignment & Cross-Platform Reliability

The web application is engineered for pixel-perfect cross-platform alignment across Windows (Edge, Chrome, Firefox), macOS (Safari, Chrome), and Linux:
- **Scrollbar Gutter Stabilization**: Configured with `scrollbar-gutter: stable; overflow-x: hidden;` preventing horizontal layout shift when vertical scrollbars appear on Windows.
- **Native Windows Typography Stack**: Uses `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif` for crisp font rendering on Windows 10/11 ClearType displays.
- **Responsive Table Wrappers**: `.table-container` wrappers ensure complex benchmark tables scroll horizontally without breaking card boundaries or blowing out flexbox grids.
- **Flexible Grid System**: Uses `min-width: 0` on CSS grid items to eliminate flex item blowout bugs in Chromium on Windows.

---

## 7. Embedded Firmware Footprint & Micro-Benchmark

| Metric | Measured Value | Implementation Note |
|:---|:---:|:---|
| **Architecture** | 3 &rarr; 16 &rarr; 1 | Inputs: $[V, I, V_{\text{smooth}}]$, $\tanh$ hidden units |
| **Total Trainable Parameters** | **81** | $W_1$ (48) + $b_1$ (16) + $W_2$ (16) + $b_2$ (1) |
| **FP32 Flash / RAM Footprint** | **348 bytes** | Weights (324 B) + Scaler constants (24 B) |
| **Multiply-Accumulates (MAC)** | **64 MACs** | $3 \times 16 + 16 \times 1$ per inference |
| **Inference Latency (Scalar)** | **6.03 &mu;s** | Evaluated on single CPU core (Xtensa portable) |
| **INT8 Quantization Impact** | **0.15 % points** | Validates 8-bit fixed-point microcontroller port |
| **C vs Python Golden Parity** | **0.000027 %** | Host GCC forward pass matches Python to $2.7 \times 10^{-5}$ |

---

## 8. Web Dashboard Features (6 Tabs)
 
 1. **Tab 1: Live Twin**:
    - Circular animated SVG SOC ring with dynamic color transitions (Green &rarr; Amber &rarr; Red).
    - **SOH Dual Aging Monitoring Card**: Real-time display of Overall SOH, Capacity Fade ($SOH_C$), Resistance Fade ($SOH_R$), Estimated $R_0$, and Remaining Useful Life (RUL).
    - Live hardware chain flow boxes with animated signal telemetry.
    - Monospace Wi-Fi JSON payload inspector with clipboard copy.
    - Interactive controls: Speed multipliers (1x to 100x), live ADC noise (0-30 mV) and current offset (-100 to +100 mA) sliders, and CSV export.
    - Multi-curve real-time charts: SOC tracking (True, ML, SPKF, EKF, CC), estimation error band, reconstructed voltage and current.
    - Live accuracy metrics: ML MAE, SPKF MAE, EKF MAE, Coulomb Counting MAE, and inference latency.
 2. **Tab 2: Evaluation**:
    - **Algorithm Recommendation Engine Banner**: Detailed verdict explaining why MLP 3-16-1 is the top-ranked choice for embedded BMS.
    - **Multi-Algorithm Comparison Scorecard**: Side-by-side metrics across MLP, SPKF, EKF, Coulomb Counting, and Linear Regression.
    - **SOH Degradation Profile**: Empirical 30-cycle NASA capacity fade and resistance growth progression table.
    - Held-out test results table with quantitative metrics across 0.5A, 1.0A, 2.0A, and 3.0A cycles.
    - Publication-grade comparative tracking plots and sensor drift divergence analysis.
 3. **Tab 3: Model Explorer**:
    - Interactive 3-16-1 neural network diagram with live neuron activations.
    - Embedded memory footprint (348 bytes) and INT8 quantization impact analysis.
    - Physical principle: Terminal voltage $IR$-drop shift curves.
    - Architecture width/depth Pareto efficiency ablation analysis.
 4. **Tab 4: Data Explorer**:
    - Visualizer for all 60 NASA cycling files.
    - Uniform 5.0s resampling interval distribution histogram.
    - 4-wire Kelvin probe contact resistance spread analysis across probes $V_{\text{bat}1} - V_{\text{bat}4}$.
 5. **Tab 5: Custom Import & Predict**:
    - User CSV upload with auto-delimiter detection and column alias mapping.
    - Vectorized pure-NumPy inference with interactive time-series plots and CSV export.
    - One-click replay into the live Digital Twin.
 6. **Tab 6: About**:
    - Academic project deliverable details and student author roster.
    - Physical equations for SOH capacity/resistance fade, EKF/SPKF formulations, and Coulomb counting derivation.

---

## 5. Known Limitations

- **Cycle vs Cell Generalisation**: The held-out test cycles originate from the same physical cells under different cycling currents. This proves generalisation across operating dynamics and rates, rather than cross-cell manufacturing variances.
- **Label Derivation**: SOC reference labels are calculated from normalized Coulomb Counting capacity per cycle, inheriting its fundamental bounds.
- **Short vs Long Run Performance**: On short, high-rate runs (2 A pair), Coulomb Counting starting from the true initial SOC remains accurate; on multi-hour runs, Coulomb Counting steadily drifts, while the ML model remains bounded.

---

## Project Team

- **Maanas S** (CB.EN.U4ELC24052)
- **Abrar M** (CB.EN.U4ELC24005)
- **Abhinav Pradeep** (CB.EN.U4ELC24004)
- **Varun Ladda** (CB.EN.U4ELC24062)

*Amrita School of Engineering, Amrita Vishwa Vidyapeetham*
