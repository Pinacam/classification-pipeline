
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include <math.h>
#include "model.h"

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

const int UV_PIN = 34;

const int SDA_PIN = 21;
const int SCL_PIN = 22;

const float ALPHA = 0.2f;                      // EMA smoothing factor
const unsigned long SAMPLE_INTERVAL_MS = 100; // ~10 Hz target

float uvFiltered = -1;
float irFiltered = -1;

unsigned long lastSampleTime = 0;


void configureTSL2591() {
  // Same settings used during data collection
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
}


// Scale sensor readings using the training bounds
float scaleValue(float value, float minimum, float maximum) {
  float scaled = (value - minimum) / (maximum - minimum);
  return constrain(scaled, 0.0f, 1.0f);
}


// Classify one pair of filtered sensor readings
int classifyLighting(float uv, float ir) {

  // Apply the same logarithmic IR transformation as Python
  float logIR = log1pf(ir);

  // Scale UV and IR using the training bounds
  float x = scaleValue(uv, SCALE_BOUNDS[0], SCALE_BOUNDS[1]);
  float y = scaleValue(logIR, SCALE_BOUNDS[2], SCALE_BOUNDS[3]);

  // Find the corresponding lookup grid cell
  int col = constrain((int)(x * GRID_N), 0, GRID_N - 1);
  int row = constrain((int)(y * GRID_N), 0, GRID_N - 1);

  int bestClass = -1;
  uint8_t bestValue = 0;

  // Compare the stored density for each lighting class
  for (int c = 0; c < NUM_CLASSES; c++) {

    uint8_t value = CLASS_TABLE[c][row][col];

    if (bestClass == -1 || value > bestValue) {
      bestValue = value;
      bestClass = c;
    }
  }

  // Reject predictions with low density
  if ((bestValue / 255.0f) < REJECT_TAU) {
    return -1;
  }

  return bestClass;
}


void setup() {
  Serial.begin(115200);

  analogReadResolution(12);

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!tsl.begin()) {
    Serial.println("ERROR: TSL2591 not found.");
    while (1) {
      delay(1000);
    }
  }

  configureTSL2591();

  Serial.println("ESP32 Lighting Classification Ready");
  Serial.println("Classes: LED, fluorescent, shade, sun");
}


void loop() {

  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;

    // Read UV sensor
    int rawUV = analogRead(UV_PIN);

    // Read TSL2591 IR sensor
    uint32_t lum = tsl.getFullLuminosity();

    uint16_t full = lum & 0xFFFF;
    uint16_t ir = lum >> 16;

    // Skip saturated readings
    if (full == 0xFFFF || ir == 0xFFFF) {
      Serial.println("WARNING: TSL2591 saturated.");
      return;
    }

    int rawIR = ir;

    // Apply the same EMA filter used during data collection
    if (uvFiltered < 0) {
      uvFiltered = rawUV;
      irFiltered = rawIR;
    } else {
      uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
      irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
    }

    // Measure inference time (excluding sensor reading)
    unsigned long startTime = micros();

    int prediction = classifyLighting(uvFiltered, irFiltered);

    unsigned long inferenceTime = micros() - startTime;

   
    // Print raw and filtered sensor readings for the report
    // These values will be used to compare EMA smoothing

    Serial.print("Raw UV: ");
    Serial.print(rawUV);

    Serial.print(" | Filtered UV: ");
    Serial.print(uvFiltered, 2);

    Serial.print(" | Raw IR: ");
    Serial.print(rawIR);

    Serial.print(" | Filtered IR: ");
    Serial.print(irFiltered, 2);

    // Print the predicted lighting condition
    Serial.print(" | Prediction: ");

    if (prediction == -1) {
        Serial.print("UNKNOWN");
    } else {
        Serial.print(CLASS_NAMES[prediction]);
    }

    // Print the time needed to classify one reading
    Serial.print(" | Inference: ");
    Serial.print(inferenceTime);
    Serial.println(" us");
  }//end sample interval
}//end of loop
