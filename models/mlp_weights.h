/*
 * Auto-generated MLP 3-16-1 Weights Header for ESP32 Battery Twin
 * Parameters: 81 (float32)
 * Inputs: [V (V), I (A), V_smooth (V)]
 */

#ifndef MLP_WEIGHTS_H
#define MLP_WEIGHTS_H

#define MLP_INPUT_DIM 3
#define MLP_HIDDEN_DIM 16
#define MLP_TOTAL_PARAMS 81

// Feature standardisation constants (fit on training set only)
static const float SCALER_MEAN[3] = {3.7799594f, -0.0849272f, 3.7801692f};
static const float SCALER_STD[3]  = {0.2959020f, 1.5045769f, 0.2942926f};

// Hidden layer weights W1 (16 neurons x 3 inputs)
static const float W1[16][3] = {
    {0.3101241f, -0.6295156f, -0.7249884f},
    {0.2824261f, 0.1346420f, 1.8851448f},
    {-0.7770783f, -2.6018915f, -1.5933526f},
    {-0.3751155f, -0.2210368f, 0.5861844f},
    {0.0115007f, -1.6134455f, -0.9561781f},
    {0.0735814f, -0.9200637f, 0.5062416f},
    {-0.7243447f, 0.0442141f, 0.8525847f},
    {-0.2332434f, -0.1618113f, -2.0718169f},
    {0.3197225f, 0.7379470f, -0.7946254f},
    {0.8126425f, -0.0848982f, 1.7304462f},
    {0.3302060f, 0.0450750f, -0.4847561f},
    {0.1417478f, 0.8269342f, 1.5309711f},
    {-0.0865780f, 2.6884179f, 0.5294740f},
    {-0.4933614f, 0.9913551f, -0.6257350f},
    {0.1356525f, -0.0519223f, -2.6691492f},
    {0.1104686f, 0.1119818f, -0.2703241f},
};

// Hidden layer bias b1 (16 neurons)
static const float B1[16] = {
    -2.0165124f, 1.7113969f, 1.3724819f, 4.3611779f, 2.7556157f, 4.2416635f, 3.1763189f, 1.3514881f, -4.4003162f, -2.3202808f, -4.2880282f, 1.3795207f, 0.8079870f, -2.6813698f, -0.2619579f, -4.4627857f, 
};

// Output layer weights W2 (16 weights)
static const float W2[16] = {
    -6.6062040f, 7.9769311f, -3.5595701f, 6.6811099f, -7.5215960f, 6.9899049f, 6.7329779f, -8.1505871f, -6.8981352f, 7.9287267f, -6.0588169f, 7.9496007f, 6.0608754f, -6.5009189f, -8.4777212f, -6.7971745f, 
};

// Output layer bias b2
static const float B2 = 6.7084127f;

#endif // MLP_WEIGHTS_H
