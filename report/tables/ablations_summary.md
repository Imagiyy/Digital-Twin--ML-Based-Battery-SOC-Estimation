# Model Ablation Studies

### 1. Architecture Ablation

| Architecture              |   Parameters |   Pair 10 MAE (%) |   Pair 10 RMSE (%) |   Latency (µs) |
|:--------------------------|-------------:|------------------:|-------------------:|---------------:|
| 4 units (tanh)            |           21 |             25.54 |              34.85 |        16120.1 |
| 8 units (tanh)            |           41 |             13.36 |              20.56 |        15158.9 |
| 16 units (tanh, Proposed) |           81 |              5.11 |               7.49 |        22245.7 |
| 32 units (tanh)           |          161 |              2.41 |               2.99 |        11028.9 |
| 64 units (tanh)           |          321 |              1.66 |               2.13 |        13246   |
| Two Layers (16-16, tanh)  |          353 |              4.14 |               7.69 |        12508.5 |
| 16 units (ReLU)           |           81 |              1.32 |               1.72 |        12441.5 |
| 16 units (Sigmoid)        |           81 |              7.18 |               9.69 |        12327   |

### 2. Feature Ablation

| Feature Set                 |   Input Dim |   Pair 10 MAE (%) |
|:----------------------------|------------:|------------------:|
| V Only                      |           1 |              9.11 |
| V + I                       |           2 |              5.43 |
| V + V_smooth                |           2 |              9.11 |
| V + I + V_smooth (Proposed) |           3 |              5.11 |

### 3. Noise Augmentation Ablation

| Condition                                     |   Pair 10 MAE (%) |
|:----------------------------------------------|------------------:|
| With Hardware Noise Augmentation (Proposed)   |              5.11 |
| Without Noise Augmentation (Clean Train Only) |             21.09 |