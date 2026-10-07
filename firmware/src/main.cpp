/*
 * ESP32 Battery Monitoring System - Firmware Implementation
 * Pairing: Virtual / Physical ESP32 + TP4056 + 18650 Li-ion Cell
 *
 * Hardware Pin Mapping:
 * - GPIO 34 (ADC1_CH6): Voltage divider (/2) input (0 - 2.1V on pin)
 * - GPIO 35 (ADC1_CH7): Current sensor analog or I2C (INA219)
 * - GPIO 18 (Input Pull-up): TP4056 CHRG pin (Active-Low)
 * - GPIO 19 (Input Pull-up): TP4056 STDBY pin (Active-Low)
 * - Built-in LED: Heartbeat
 */

#include <Arduino.h>
#include <WiFi.h>
#include "mlp.h"

// Configuration Constants
#define ADC_PIN              34
#define CURRENT_PIN          35
#define TP4056_CHRG_PIN      18
#define TP4056_STDBY_PIN     19

#define SAMPLE_INTERVAL_MS   5000
#define SMOOTH_WINDOW_SIZE   12
#define DIVIDER_RATIO        0.5f
#define ADC_VREF             3.3f
#define ADC_MAX_CODE         4095.0f

// Ring buffer for causal 60s moving average (12 samples @ 5s)
static float v_ring_buffer[SMOOTH_WINDOW_SIZE];
static int ring_head = 0;
static int samples_collected = 0;
static unsigned long last_sample_time = 0;

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n[ESP32] Initializing ML SOC Digital Twin Firmware...");

    // ADC Configuration: 12-bit resolution, 11dB attenuation for 0-3.3V range
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    // TP4056 LED input status pins
    pinMode(TP4056_CHRG_PIN, INPUT_PULLUP);
    pinMode(TP4056_STDBY_PIN, INPUT_PULLUP);

    // Initialize ring buffer
    for (int i = 0; i < SMOOTH_WINDOW_SIZE; i++) {
        v_ring_buffer[i] = 0.0f;
    }

    Serial.println("[ESP32] Hardware initialization complete. Streaming telemetry:");
}

float read_cell_voltage() {
    // Read raw 12-bit code (0-4095)
    int adc_raw = analogRead(ADC_PIN);
    // Convert to pin voltage using Vref = 3.3V
    float v_pin = ((float)adc_raw / ADC_MAX_CODE) * ADC_VREF;
    // Reconstruct cell voltage using divider ratio (/2)
    float v_cell = v_pin / DIVIDER_RATIO;
    return v_cell;
}

float read_cell_current() {
    // Simulated reading or analog/INA219 current reading
    // Discharge positive convention: discharge > 0, charge < 0
    return 1.00f; // placeholder
}

String get_tp4056_status() {
    bool chrg_active = (digitalRead(TP4056_CHRG_PIN) == LOW);
    bool stdby_active = (digitalRead(TP4056_STDBY_PIN) == LOW);

    if (chrg_active && !stdby_active) {
        return "CHARGING";
    } else if (!chrg_active && stdby_active) {
        return "CHARGED_STANDBY";
    } else {
        return "DISCHARGING";
    }
}

void loop() {
    unsigned long current_time = millis();
    if (current_time - last_sample_time >= SAMPLE_INTERVAL_MS) {
        last_sample_time = current_time;

        // 1. Read calibrated hardware inputs
        float v_cell = read_cell_voltage();
        float i_cell = read_cell_current();
        String status = get_tp4056_status();

        // 2. Update Causal Moving Average Ring Buffer (Expanding window warm-up)
        v_ring_buffer[ring_head] = v_cell;
        ring_head = (ring_head + 1) % SMOOTH_WINDOW_SIZE;
        if (samples_collected < SMOOTH_WINDOW_SIZE) {
            samples_collected++;
        }

        float v_smooth_acc = 0.0f;
        for (int k = 0; k < samples_collected; k++) {
            v_smooth_acc += v_ring_buffer[k];
        }
        float v_smooth = v_smooth_acc / (float)samples_collected;

        // 3. Run On-Chip MLP 3-16-1 Inference
        float soc_estimate = mlp_predict(v_cell, i_cell, v_smooth);

        // 4. Stream Telemetry JSON Payload over Serial and Wi-Fi
        Serial.printf("{\"v\": %.3f, \"i\": %.2f, \"soc\": %.1f, \"status\": \"%s\"}\n",
                      v_cell, i_cell, soc_estimate, status.c_str());
    }
}
