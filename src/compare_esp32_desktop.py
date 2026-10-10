
"""
Part 4 -- Compare ESP32 predictions with the Python lookup model.

Uses real UV and IR readings collected from the ESP32.
The same measurements are passed through the Python model
to check if both systems produce the same classification.
"""

import numpy as np

from pipeline_common import (
    load_split_scaled,
    scale_features,
    cell_of,
    quantize,
    REJECT_TAU
)


# Live sensor readings collected from the ESP32
# Each sample contains: UV, IR, ESP32 prediction
samples = [
    # Shade
    (0.00, 167.34, "shade"),
    (0.00, 167.60, "shade"),
    (0.00, 167.94, "shade"),

    # LED
    (0.00, 199.33, "LED"),
    (0.00, 201.65, "LED"),
    (0.00, 218.50, "LED"),

    # Fluorescent
    (0.00, 468.94, "fluorescent"),
    (0.00, 490.40, "fluorescent"),
    (0.00, 513.87, "fluorescent"),

    # Sunlight
    (4095.00, 37889.00, "sun")
]


# Load the training data and scaling bounds
# These are the same bounds used to generate model.h
data = load_split_scaled()
bounds = data["bounds"]


# Load the KDE surfaces generated during training
surfaces = np.load("surfaces.npz")


# Class names in the same order as the ESP32 model
class_names = [
    "LED",
    "fluorescent",
    "shade",
    "sun"
]


# Convert each KDE surface into uint8 values
# This reproduces the lookup tables stored in model.h
tables = [
    quantize(surfaces[f"c{i}"])
    for i in range(4)
]


# Classify one pair of UV and IR measurements
def predict(uv, ir):

    # Combine the two sensor readings
    point = np.array([[uv, ir]])

    # Apply logarithmic IR scaling and min-max scaling
    # using the original training bounds
    scaled, _ = scale_features(point, bounds)

    # Find the corresponding row and column in the grid
    row, col = cell_of(scaled[0])

    # Read the stored density for each lighting class
    values = [
        table[row, col] / 255.0
        for table in tables
    ]

    # Find the class with the highest density
    best = int(np.argmax(values))

    # Return UNKNOWN if the density is too low
    if values[best] < REJECT_TAU:
        return "UNKNOWN"

    # Return the predicted lighting condition
    return class_names[best]


# Compare the ESP32 and Python predictions
correct = 0

for uv, ir, esp_prediction in samples:

    # Run the same sensor readings through Python
    python_prediction = predict(uv, ir)

    # Check whether both models agree
    match = python_prediction == esp_prediction

    if match:
        correct += 1

    # Print the results for each sample
    print(
        f"ESP32: {esp_prediction:<12} "
        f"Python: {python_prediction:<12} "
        f"Match: {match}"
    )


# Calculate the percentage of matching predictions
agreement = 100 * correct / len(samples)

# Print the final results
print(f"\nSamples compared: {len(samples)}")
print(f"Matching predictions: {correct}")
print(f"ESP32 vs Python agreement: {agreement:.1f}%")
