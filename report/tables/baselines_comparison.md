# Baselines Comparison Table

| Cycle                | Model                              |   MAE (%) |   RMSE (%) |   Max Error (%) |
|:---------------------|:-----------------------------------|----------:|-----------:|----------------:|
| Discharge 02 (0.5 A) | MLP 3-16-1 (Proposed)              |      1.28 |       1.69 |            5.46 |
| Discharge 02 (0.5 A) | Random Forest                      |      2.25 |       2.59 |            7.65 |
| Discharge 02 (0.5 A) | Linear Regression                  |      2.53 |       3.3  |           12.29 |
| Discharge 02 (0.5 A) | SPKF (Sigma-Point Kalman)          |      4.84 |       5.39 |            8.88 |
| Discharge 02 (0.5 A) | EKF (Extended Kalman)              |      4.84 |       5.39 |            8.91 |
| Discharge 02 (0.5 A) | Coulomb Counting (True SOC0)       |      3.32 |       3.83 |            6.64 |
| Discharge 02 (0.5 A) | Coulomb Counting (Wrong SOC0 -10%) |     13.32 |      13.46 |           16.64 |
| Discharge 02 (0.5 A) | OCV Table Lookup (V-only)          |      2.54 |       3.17 |           12.94 |
| Pair 10 (1.0 A)      | MLP 3-16-1 (Proposed)              |      1.16 |       1.57 |            6.19 |
| Pair 10 (1.0 A)      | Random Forest                      |      3.3  |       4.13 |           12.03 |
| Pair 10 (1.0 A)      | Linear Regression                  |      2.68 |       3.32 |            9.85 |
| Pair 10 (1.0 A)      | SPKF (Sigma-Point Kalman)          |      4.95 |       5.9  |           10.73 |
| Pair 10 (1.0 A)      | EKF (Extended Kalman)              |      4.95 |       5.9  |           10.73 |
| Pair 10 (1.0 A)      | Coulomb Counting (True SOC0)       |      2.37 |       2.67 |            4.26 |
| Pair 10 (1.0 A)      | Coulomb Counting (Wrong SOC0 -10%) |      7.63 |       7.73 |           10    |
| Pair 10 (1.0 A)      | OCV Table Lookup (V-only)          |      8.4  |      11.78 |           26.67 |
| Pair 20 (2.0 A)      | MLP 3-16-1 (Proposed)              |      1.07 |       1.49 |           11.05 |
| Pair 20 (2.0 A)      | Random Forest                      |      1.11 |       1.6  |           11.02 |
| Pair 20 (2.0 A)      | Linear Regression                  |      3.06 |       3.76 |           10.52 |
| Pair 20 (2.0 A)      | SPKF (Sigma-Point Kalman)          |      5.77 |       7.31 |           14.43 |
| Pair 20 (2.0 A)      | EKF (Extended Kalman)              |      5.77 |       7.3  |           14.41 |
| Pair 20 (2.0 A)      | Coulomb Counting (True SOC0)       |      0.95 |       1.04 |            1.51 |
| Pair 20 (2.0 A)      | Coulomb Counting (Wrong SOC0 -10%) |      9.05 |       9.06 |           10    |
| Pair 20 (2.0 A)      | OCV Table Lookup (V-only)          |     12.9  |      15.82 |           29.74 |
| Pair 30 (3.0 A)      | MLP 3-16-1 (Proposed)              |      1.49 |       2.12 |           11.29 |
| Pair 30 (3.0 A)      | Random Forest                      |      1.5  |       2.06 |           11.23 |
| Pair 30 (3.0 A)      | Linear Regression                  |      3.81 |       4.65 |           11.85 |
| Pair 30 (3.0 A)      | SPKF (Sigma-Point Kalman)          |      8.5  |      10.27 |           17.71 |
| Pair 30 (3.0 A)      | EKF (Extended Kalman)              |      8.51 |      10.27 |           17.67 |
| Pair 30 (3.0 A)      | Coulomb Counting (True SOC0)       |      2.33 |       2.59 |            3.61 |
| Pair 30 (3.0 A)      | Coulomb Counting (Wrong SOC0 -10%) |      7.67 |       7.75 |           10    |
| Pair 30 (3.0 A)      | OCV Table Lookup (V-only)          |     15.18 |      17.92 |           34.88 |