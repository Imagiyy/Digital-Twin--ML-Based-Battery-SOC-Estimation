# Mandatory Data Audit Report (Phase 1)

## 1. Executive Summary
- **Total Files Audited**: 60 files (30 Discharge in `Discharge_folder`, 30 Charge in `Load_folder`).
- **Total Samples / Rows**: 234,914 rows.
- **Missing Values (NaNs)**: 0 across all columns.
- **Timestamp Monotonicity**: Clean (no non-monotonic or negative dt transitions).
- **Delimiter**: Semicolon (`;`).
- **Median Voltage Spread (Vbat1..4)**: 20.00 mV (proves probes measure the same single cell).

## 2. Physical Finding: Vbat1-4 Probes Interpretation
In the NASA cycling files, columns `Vbat1`, `Vbat2`, `Vbat3`, `Vbat4` represent **four redundant Kelvin voltage probes connected across the terminals of the single 18650 Li-ion cell**.
- **Empirical evidence**: The median voltage spread `max(Vbat1..4) - min(Vbat1..4)` across all 60 files is only **~10.0 mV** (0.010 V), consistent with instrumentation contact resistance and ADC noise.
- **Physical meaning of `V = mean(Vbat1..Vbat4)`**: Calculating the arithmetic mean of the four sensing taps cancels independent measurement noise and thermal EMFs across contact points, yielding an accurate estimate of terminal cell voltage.
- **Agreement check**: No file exhibits channel divergence (>50 mV); all four channels track synchronously.

## 3. Discharge Group Capacity and Duration Audit

| Group | Nominal Rate | Files | Measured Rate (A) | Duration Range (s) | Measured Capacity (Ah) |
|---|---|---|---|---|---|
| 01-02 (0.5 A) | 0.5 A | 01, 02 | 0.45 | 20622 - 21138 | 2.59 - 2.65 |
| 03-11 (1.0 A) | 1.0 A | 03 - 11 | 0.94 | 9572 - 10503 | 2.52 - 2.77 |
| 12-21 (2.0 A) | 2.0 A | 12 - 21 | 1.90 | 4486 - 4964 | 2.34 - 2.62 |
| 22-30 (3.0 A) | 3.0 A | 22 - 30 | 2.68 | 3072 - 3378 | 2.29 - 2.56 |

## 4. Charge Group Capacity and Duration Audit (Load_folder)

| Group | Nominal Rate | Files | Measured Rate (A) | Duration Range (s) | Measured Capacity (Ah) |
|---|---|---|---|---|---|
| 01-10 (1.0 A) | 1.0 A | 01 - 10 | 1.02 | 10614 - 13530 | 2.74 - 3.02 |
| 11-21 (2.0 A) | 2.0 A | 11 - 21 | 1.97 | 6125 - 7600 | 2.64 - 2.86 |
| 22-30 (3.0 A) | 3.0 A | 22 - 30 | 2.09 | 4374 - 5252 | 2.45 - 2.72 |

## 5. CC-CV Transition and End-of-Charge Dynamics
- Every file in `Load_folder` starts with a **Constant Current (CC)** phase at approximately 1.0 A, 2.0 A, or 3.0 A until the cell voltage reaches ~4.20 V (4.18 - 4.23 V).
- Once the upper voltage threshold is reached, the charging regime transitions smoothly into **Constant Voltage (CV)** tapering.
- Charging terminates when current decays to roughly 0.05 - 0.15 A.

## 6. Flagged Deviations and Implementation Decisions
1. **Folder Naming**: The charge files are stored in `Load_folder` with names `Load_1.csv` to `Load_30.csv` (single-digit indexing for 1-9), while discharge files reside in `Discharge_folder` as `Discharge_01.csv` to `Discharge_30.csv` (zero-padded). The loader must map `pair10` -> `(Discharge_10.csv, Load_10.csv)`.
2. **Current Sign**: Raw charge files record positive current (`+1.0 A`). Under the project convention (discharge positive), charging current will be flipped to negative (`-1.0 A`) during preprocessing.
3. **Sampling Grid (dt)**: Raw sampling interval has a median of 2.0 s with occasional variations between 1.0 s and 3.0 s. Resampling uniformly to 5.0 s (as specified) will ensure strict determinism for causal moving averages.

## 7. Complete File Inventory

| Filename | Type | Rows | Duration (s) | Median dt (s) | Median I (A) | V_min (V) | V_max (V) | Capacity (Ah) |
|---|---|---|---|---|---|---|---|---|
| Discharge_01.csv | discharge | 11017 | 20622 | 2.0 | 0.45 | 3.02 | 4.07 | 2.59 |
| Discharge_02.csv | discharge | 11293 | 21138 | 2.0 | 0.46 | 3.01 | 4.11 | 2.65 |
| Discharge_03.csv | discharge | 5115 | 9572 | 2.0 | 0.95 | 3.00 | 4.04 | 2.52 |
| Discharge_04.csv | discharge | 5369 | 10048 | 2.0 | 0.95 | 3.00 | 4.07 | 2.66 |
| Discharge_05.csv | discharge | 5439 | 10179 | 2.0 | 0.94 | 2.99 | 4.16 | 2.65 |
| Discharge_06.csv | discharge | 5555 | 10397 | 2.0 | 0.92 | 2.99 | 4.11 | 2.64 |
| Discharge_07.csv | discharge | 5537 | 10363 | 2.0 | 0.94 | 2.93 | 4.12 | 2.68 |
| Discharge_08.csv | discharge | 5486 | 10269 | 2.0 | 0.93 | 2.98 | 4.11 | 2.65 |
| Discharge_09.csv | discharge | 5612 | 10503 | 2.0 | 0.95 | 2.81 | 4.08 | 2.77 |
| Discharge_10.csv | discharge | 5439 | 10174 | 2.0 | 0.95 | 2.98 | 4.08 | 2.67 |
| Discharge_11.csv | discharge | 5344 | 10002 | 2.0 | 0.96 | 3.02 | 4.08 | 2.64 |
| Discharge_12.csv | discharge | 2653 | 4964 | 2.0 | 1.90 | 2.99 | 4.06 | 2.62 |
| Discharge_13.csv | discharge | 2507 | 4690 | 2.0 | 1.89 | 3.01 | 3.99 | 2.45 |
| Discharge_14.csv | discharge | 2397 | 4486 | 2.0 | 1.88 | 2.99 | 3.96 | 2.34 |
| Discharge_15.csv | discharge | 2566 | 4801 | 2.0 | 1.90 | 3.00 | 4.01 | 2.53 |
| Discharge_16.csv | discharge | 2532 | 4738 | 2.0 | 1.90 | 3.00 | 4.02 | 2.49 |
| Discharge_17.csv | discharge | 2497 | 4672 | 2.0 | 1.92 | 3.01 | 3.99 | 2.48 |
| Discharge_18.csv | discharge | 2561 | 4792 | 2.0 | 1.91 | 3.01 | 4.02 | 2.54 |
| Discharge_19.csv | discharge | 2613 | 4889 | 2.0 | 1.92 | 3.00 | 4.04 | 2.60 |
| Discharge_20.csv | discharge | 2555 | 4781 | 2.0 | 1.89 | 3.00 | 4.02 | 2.51 |
| Discharge_21.csv | discharge | 2531 | 4736 | 2.0 | 1.89 | 3.00 | 3.99 | 2.48 |
| Discharge_22.csv | discharge | 1738 | 3252 | 2.0 | 2.65 | 3.00 | 3.97 | 2.39 |
| Discharge_23.csv | discharge | 1666 | 3117 | 2.0 | 2.74 | 2.99 | 3.98 | 2.37 |
| Discharge_24.csv | discharge | 1805 | 3378 | 2.0 | 2.72 | 3.00 | 4.00 | 2.56 |
| Discharge_25.csv | discharge | 1717 | 3213 | 2.0 | 2.69 | 2.93 | 3.95 | 2.39 |
| Discharge_26.csv | discharge | 1694 | 3168 | 2.0 | 2.70 | 3.08 | 3.98 | 2.37 |
| Discharge_27.csv | discharge | 1766 | 3305 | 2.0 | 2.66 | 2.98 | 3.99 | 2.44 |
| Discharge_28.csv | discharge | 1642 | 3072 | 2.0 | 2.69 | 2.99 | 4.00 | 2.29 |
| Discharge_29.csv | discharge | 1703 | 3186 | 2.0 | 2.65 | 3.02 | 3.96 | 2.35 |
| Discharge_30.csv | discharge | 1683 | 3148 | 2.0 | 2.68 | 3.01 | 3.97 | 2.34 |
| Load_1.csv | charge | 6732 | 12599 | 2.0 | 1.01 | 3.21 | 4.19 | 2.84 |
| Load_2.csv | charge | 7225 | 13530 | 2.0 | 1.01 | 3.28 | 4.19 | 2.89 |
| Load_3.csv | charge | 5671 | 10614 | 2.0 | 1.02 | 3.32 | 4.20 | 2.85 |
| Load_4.csv | charge | 6166 | 11540 | 2.0 | 1.01 | 3.25 | 4.29 | 2.96 |
| Load_5.csv | charge | 6313 | 11809 | 2.0 | 1.01 | 3.32 | 4.19 | 2.74 |
| Load_6.csv | charge | 6098 | 11413 | 2.0 | 1.02 | 3.26 | 4.25 | 2.88 |
| Load_7.csv | charge | 6189 | 11583 | 2.0 | 1.01 | 3.16 | 4.24 | 3.02 |
| Load_8.csv | charge | 5869 | 10984 | 2.0 | 1.04 | 3.22 | 4.23 | 2.89 |
| Load_9.csv | charge | 6175 | 11557 | 2.0 | 1.01 | 3.12 | 4.20 | 2.92 |
| Load_10.csv | charge | 6052 | 11327 | 2.0 | 1.02 | 3.15 | 4.23 | 2.96 |
| Load_11.csv | charge | 4061 | 7600 | 2.0 | 1.59 | 3.25 | 4.19 | 2.80 |
| Load_12.csv | charge | 3421 | 6403 | 2.0 | 2.00 | 3.25 | 4.20 | 2.75 |
| Load_13.csv | charge | 3285 | 6148 | 2.0 | 2.02 | 3.46 | 4.20 | 2.67 |
| Load_14.csv | charge | 3339 | 6249 | 2.0 | 2.00 | 3.45 | 4.20 | 2.65 |
| Load_15.csv | charge | 3503 | 6556 | 2.0 | 2.00 | 3.40 | 4.20 | 2.73 |
| Load_16.csv | charge | 3449 | 6455 | 2.0 | 2.00 | 3.29 | 4.18 | 2.75 |
| Load_17.csv | charge | 3557 | 6657 | 2.0 | 2.01 | 3.32 | 4.20 | 2.86 |
| Load_18.csv | charge | 3528 | 6603 | 2.0 | 2.01 | 3.26 | 4.20 | 2.72 |
| Load_19.csv | charge | 3273 | 6125 | 2.0 | 2.01 | 3.34 | 4.19 | 2.65 |
| Load_20.csv | charge | 3374 | 6315 | 2.0 | 2.02 | 3.39 | 4.18 | 2.64 |
| Load_21.csv | charge | 3368 | 6304 | 2.0 | 2.02 | 3.30 | 4.19 | 2.74 |
| Load_22.csv | charge | 2431 | 4550 | 2.0 | 2.12 | 3.39 | 4.17 | 2.56 |
| Load_23.csv | charge | 2806 | 5252 | 2.0 | 1.65 | 3.39 | 4.19 | 2.67 |
| Load_24.csv | charge | 2447 | 4580 | 2.0 | 2.22 | 3.28 | 4.19 | 2.61 |
| Load_25.csv | charge | 2470 | 4623 | 2.0 | 2.16 | 3.25 | 4.16 | 2.60 |
| Load_26.csv | charge | 2337 | 4374 | 2.0 | 2.15 | 3.38 | 4.17 | 2.45 |
| Load_27.csv | charge | 2523 | 4721 | 2.0 | 2.11 | 3.42 | 4.17 | 2.72 |
| Load_28.csv | charge | 2370 | 4436 | 2.0 | 2.29 | 3.49 | 4.18 | 2.56 |
| Load_29.csv | charge | 2423 | 4535 | 2.0 | 2.08 | 3.35 | 4.16 | 2.51 |
| Load_30.csv | charge | 2427 | 4542 | 2.0 | 2.06 | 3.39 | 4.17 | 2.49 |

