/*
 * Embedded C Implementation of MLP 3-16-1 Forward Pass
 * Designed for ESP32 Xtensa Dual-Core 240MHz Microcontroller
 */

#ifndef MLP_H
#define MLP_H

#include <math.h>
#include "mlp_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run MLP 3-16-1 forward pass on calibrated float32 inputs.
 *
 * @param v        Terminal cell voltage (V)
 * @param i        Current (A, discharge positive)
 * @param v_smooth Causal 60s moving average voltage (V)
 * @return         Estimated State of Charge clipped to [0.0, 100.0] %
 */
static inline float mlp_predict(float v, float i, float v_smooth) {
    // 1. Feature Standardisation
    float x0 = (v - SCALER_MEAN[0]) / SCALER_STD[0];
    float x1 = (i - SCALER_MEAN[1]) / SCALER_STD[1];
    float x2 = (v_smooth - SCALER_MEAN[2]) / SCALER_STD[2];

    // 2. Hidden Layer (16 units with tanhf activation)
    float soc_acc = B2;
    for (int j = 0; j < MLP_HIDDEN_DIM; j++) {
        float z = B1[j] + (W1[j][0] * x0) + (W1[j][1] * x1) + (W1[j][2] * x2);
        float a = tanhf(z);
        soc_acc += W2[j] * a;
    }

    // 3. Output Clipping to [0.0, 100.0]
    if (soc_acc < 0.0f) {
        return 0.0f;
    } else if (soc_acc > 100.0f) {
        return 100.0f;
    }
    return soc_acc;
}

#ifdef __cplusplus
}
#endif

#endif // MLP_H
