# Held-Out Test Evaluation Results (With 5 mV ADC Noise & 30 mA Current Offset)

| Cycle                |   Rate (A) |   ML MAE (%) |   ML RMSE (%) |   ML Max (%) |   CC MAE (%) |   CC RMSE (%) |   CC Max (%) |   ML MAE (Dis) |   ML MAE (Chg) |
|:---------------------|-----------:|-------------:|--------------:|-------------:|-------------:|--------------:|-------------:|---------------:|---------------:|
| Discharge 02 (0.5 A) |        0.5 |         1.28 |          1.69 |         5.46 |         3.32 |          3.83 |         6.64 |           1.28 |           0    |
| Pair 10 (1.0 A Pair) |        1   |         1.16 |          1.57 |         6.19 |         2.37 |          2.67 |         4.26 |           1.21 |           1.14 |
| Pair 20 (2.0 A Pair) |        2   |         1.07 |          1.49 |        11.05 |         0.95 |          1.04 |         1.51 |           0.94 |           1.04 |
| Pair 30 (3.0 A Pair) |        3   |         1.49 |          2.12 |        11.29 |         2.33 |          2.59 |         3.61 |           0.81 |           1.8  |

*Target: Mean Absolute Error < 2-3 percentage points on every cycle.*
