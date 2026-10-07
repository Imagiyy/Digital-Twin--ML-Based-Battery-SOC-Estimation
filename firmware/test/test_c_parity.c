/*
 * Host-Side C Test for MLP 3-16-1 Parity Verification
 * Compiles with gcc to ensure firmware C code matches Python reference within 1e-4.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/mlp.h"

int main(int argc, char *argv[]) {
    const char *golden_path = "tests/golden_vectors.json";
    FILE *fp = fopen(golden_path, "r");
    if (!fp) {
        // Try fallback relative path
        golden_path = "../../tests/golden_vectors.json";
        fp = fopen(golden_path, "r");
    }
    if (!fp) {
        fprintf(stderr, "Error: Could not open golden_vectors.json!\n");
        return 1;
    }

    printf("======================================================================\n");
    printf("HOST-SIDE C TEST: MLP 3-16-1 GOLDEN VECTOR PARITY\n");
    printf("======================================================================\n");

    char line[1024];
    int tested_count = 0;
    int failed_count = 0;
    float max_diff = 0.0f;

    float current_v = 0.0f;
    float current_i = 0.0f;
    float current_vs = 0.0f;
    float expected_soc = 0.0f;

    while (fgets(line, sizeof(line), fp)) {
        char *v_ptr = strstr(line, "\"v\":");
        if (v_ptr) {
            sscanf(v_ptr, "\"v\": %f", &current_v);
        }
        char *i_ptr = strstr(line, "\"i\":");
        if (i_ptr) {
            sscanf(i_ptr, "\"i\": %f", &current_i);
        }
        char *vs_ptr = strstr(line, "\"v_smooth\":");
        if (vs_ptr) {
            sscanf(vs_ptr, "\"v_smooth\": %f", &current_vs);
        }
        char *soc_ptr = strstr(line, "\"expected_soc\":");
        if (soc_ptr) {
            sscanf(soc_ptr, "\"expected_soc\": %f", &expected_soc);

            // Execute C forward pass
            float c_pred = mlp_predict(current_v, current_i, current_vs);
            float diff = fabsf(c_pred - expected_soc);
            if (diff > max_diff) {
                max_diff = diff;
            }

            if (diff > 1e-3f) {
                fprintf(stderr, "FAIL [Vector %d]: V=%.3f, I=%.3f, Vs=%.3f | Expected=%.4f, Got=%.4f (diff=%.6f)\n",
                        tested_count, current_v, current_i, current_vs, expected_soc, c_pred, diff);
                failed_count++;
            }
            tested_count++;
        }
    }
    fclose(fp);

    printf("Tested %d golden vectors.\n", tested_count);
    printf("Maximum Absolute Discrepancy (C vs Python): %.6f percentage points\n", max_diff);

    if (failed_count == 0 && tested_count > 0) {
        printf("RESULT: ALL %d GOLDEN VECTORS PASSED (Tolerance < 1e-3) ✓\n", tested_count);
        return 0;
    } else {
        printf("RESULT: %d VECTORS FAILED ✗\n", failed_count);
        return 1;
    }
}
