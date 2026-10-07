# Model Ablation Studies

### 1. Architecture Ablation

| Architecture              |   Parameters |   Pair 10 MAE (%) |   Pair 10 RMSE (%) |   Latency (µs) |
|:--------------------------|-------------:|------------------:|-------------------:|---------------:|
| 4 units (tanh)            |           21 |              5.92 |               9.9  |           6.01 |
| 8 units (tanh)            |           41 |              1.91 |               2.33 |           6.55 |
| 16 units (tanh, Proposed) |           81 |              1.19 |               1.53 |           6.8  |
| 32 units (tanh)           |          161 |              1.11 |               1.48 |           5.87 |
| 64 units (tanh)           |          321 |              1.11 |               1.48 |           7.15 |
| Two Layers (16-16, tanh)  |          353 |              0.73 |               0.93 |           7.13 |
| 16 units (ReLU)           |           81 |              1.12 |               1.52 |           6.29 |
| 16 units (Sigmoid)        |           81 |              1.29 |               1.71 |           6.24 |

### 2. Feature Ablation

| Feature Set                 |   Input Dim |   Pair 10 MAE (%) |
|:----------------------------|------------:|------------------:|
| V Only                      |           1 |              7.32 |
| V + I                       |           2 |              1.48 |
| V + V_smooth                |           2 |              6.66 |
| V + I + V_smooth (Proposed) |           3 |              1.19 |

### 3. Noise Augmentation Ablation

| Condition                                     |   Pair 10 MAE (%) |
|:----------------------------------------------|------------------:|
| With Hardware Noise Augmentation (Proposed)   |              1.19 |
| Without Noise Augmentation (Clean Train Only) |              6.48 |