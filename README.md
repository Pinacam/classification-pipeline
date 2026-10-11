
# On-Device Lighting Classification Using ESP32

## Project Overview

This project uses an ESP32 to classify four lighting conditions:
- LED
- Fluorescent
- Shade
- Sunlight

The system uses a UV sensor and a TSL2591 light sensor to collect measurements.

The classification model is trained in Python and then stored on the ESP32 as small lookup tables.

## Hardware

- ESP32 development board
- GUVA-S12SD UV sensor
- TSL2591 light sensor
- Breadboard and jumper wires

## How It Works

1. Collect UV and IR sensor readings.
2. Use DBSCAN to identify four clusters.
3. Apply KDE to model each cluster.
4. Convert the KDE results into 16 × 16 lookup tables.
5. Generate `model.h` and use it on the ESP32.
6. Apply EMA filtering to live sensor readings.
7. Display the predicted lighting condition in Arduino Serial Monitor.

## Results

- Total samples: 2,400
- DBSCAN clusters: 4
- Test accuracy: 100% (480/480 same-session held-out samples)
- ESP32/Python agreement: 100% (10/10 selected measurements)
- Lookup table storage: 1,024 bytes
- Flash usage: 308,800 bytes (23%)
- SRAM usage: 23,740 bytes (7%)
- Inference time: approximately 3–6 microseconds

## Repository Structure

- `data/` — Sensor measurements and test results
- `src/` — Python training, evaluation, and graphing scripts
- `firmware/` — ESP32 Arduino code and generated model
- `figures/generated/` — Classification and EMA graphs
- `demo/` — ESP32 demonstration video
- `Lab3.pdf` — Final project report

## Running the ESP32 Classifier

1. Open `firmware/classification/classification.ino` in Arduino IDE.
2. Select **ESP32 Dev Module** and the correct COM port.
3. Connect the UV sensor to GPIO34.
4. Connect the TSL2591 using SDA on GPIO21 and SCL on GPIO22.
5. Upload the sketch to the ESP32.
6. Open Serial Monitor at **115200 baud** to view predictions.

## Limitations

The UV sensor can reach its maximum ADC reading under strong sunlight. The model was also evaluated mainly using measurements from the same collection sessions, so performance under new conditions may differ.
