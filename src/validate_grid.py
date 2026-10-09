
"""
Validate the stored KDE lookup table before final testing.

Uses the validation dataset to check whether the 16 x 16 grid
can distinguish the four lighting conditions.
"""

import numpy as np
import pandas as pd

from pipeline_common import (
    load_split_scaled,
    cell_of,
    quantize,
    REJECT_TAU
)


# Load training and validation data
data = load_split_scaled()

Xval = data["val"]
y_val = data["y_val"]
y_train = data["y_train"]


# Load DBSCAN training labels
labels = np.loadtxt("labels.csv", dtype=int)

classes = sorted(set(labels) - {-1})


# Match discovered clusters to lighting names
confusion = pd.crosstab(
    pd.Series(y_train, name="Actual"),
    pd.Series(labels, name="Cluster")
)

cluster_names = {
    c: confusion[c].idxmax()
    for c in classes
}


# Load the stored KDE surfaces
surfaces = np.load("surfaces.npz")

tables = {
    c: quantize(surfaces[f"c{c}"])
    for c in classes
}


# Classify one validation measurement
def classify(point):

    row, col = cell_of(point)

    values = np.array([
        tables[c][row, col] / 255.0
        for c in classes
    ])

    best = int(np.argmax(values))

    if values[best] < REJECT_TAU:
        return "unknown"

    return cluster_names[classes[best]]


# Predict lighting conditions
predictions = np.array([
    classify(point)
    for point in Xval
])


# Calculate validation accuracy
accuracy = (predictions == y_val).mean()

print(f"Validation samples: {len(Xval)}")
print(f"Grid size: {tables[classes[0]].shape}")
print(f"Validation accuracy: {100 * accuracy:.1f}%")


# Compare actual and predicted lighting
results = pd.crosstab(
    pd.Series(y_val, name="Actual"),
    pd.Series(predictions, name="Predicted")
)

print("\nValidation Confusion Table:")
print(results)
