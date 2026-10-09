
"""
Part 4 -- spend the TEST set once, on the frozen design.

Compare the continuous KDE model with the stored uint8 model
using real UV and IR sensor data.

The actual lighting labels are also used to measure how well
the stored model classifies LED, fluorescent, shade, and sun.
"""


import numpy as np
import pandas as pd

from pipeline_common import (
    load_split_scaled,
    make_grid,
    cell_of,
    fit_density,
    normalise_peak,
    quantize,
    dequantize,
    REJECT_TAU
)


# Frozen validation-selected bandwidth
BANDWIDTH = 0.03


# Load data
data = load_split_scaled()

Xtr = data["train"]
Xte = data["test"]
y_test = data["y_test"]


# DBSCAN labels from Part 1
labels = np.loadtxt(
    "labels.csv",
    dtype=int
)

classes = sorted(
    set(labels) - {-1}
)


# Lookup grid
xx, yy, grid_points = make_grid()

# Continuous KDE model

kdes = {}
peaks = {}

for c in classes:

    kde = fit_density(
        Xtr[labels == c],
        BANDWIDTH
    )

    kdes[c] = kde

    peaks[c] = np.exp(
        kde.score_samples(
            grid_points
        )
    ).max()


# Stored uint8 model

surfaces = np.load(
    "surfaces.npz"
)

tables = {
    c: quantize(
        surfaces[f"c{c}"]
    )
    for c in classes
}

# -1 means unknown / rejected
LABELS = classes + [-1]

# Continuous decision

def continuous_decision(point):

    values = np.array([
        np.exp(
            kdes[c].score_samples(
                [point]
            )[0]
        ) / peaks[c]
        for c in classes
    ])

    best = int(
        np.argmax(values)
    )

    if values[best] < REJECT_TAU:
        return -1

    return classes[best]

# Stored decision

def stored_decision(point):

    r, c = cell_of(point)

    values = np.array([
        dequantize(
            tables[k][r, c]
        )
        for k in classes
    ])

    best = int(
        np.argmax(values)
    )

    if values[best] < REJECT_TAU:
        return -1

    return classes[best]

# Spend test set once

cont = np.array([
    continuous_decision(p)
    for p in Xte
])

stor = np.array([
    stored_decision(p)
    for p in Xte
])


agree = (
    cont == stor
).mean()


print(
    f"test points: {len(Xte)}"
)

print(
    f"stored-vs-continuous agreement "
    f"on held-out test: "
    f"{100 * agree:.1f}%"
)

# Confusion table

print(
    "\nconfusion "
    "(rows = continuous KDE, "
    "cols = stored uint8):"
)


head = "".join(
    f"{('unk' if label == -1 else 'c' + str(label)):>6}"
    for label in LABELS
)

print(
    f"{'':>10}{head}"
)


for true_label in LABELS:

    row = "".join(
        f"{int(((cont == true_label) & (stor == predicted)).sum()):>6}"
        for predicted in LABELS
    )

    name = (
        "unknown"
        if true_label == -1
        else f"class {true_label}"
    )

    print(
        f"{name:>10}{row}"
    )

    
# Compare stored predictions with actual lighting labels

# Match DBSCAN cluster numbers to actual lighting classes
training_confusion = pd.crosstab(
    pd.Series(data["y_train"], name="Actual"),
    pd.Series(labels, name="Cluster")
)

cluster_names = {}

for c in classes:
    cluster_names[c] = training_confusion[c].idxmax()

# Convert stored cluster predictions into lighting names
predicted_names = np.array([
    cluster_names[c] if c != -1 else "unknown"
    for c in stor
])

# Calculate classification accuracy
accuracy = (predicted_names == y_test).mean()

print(
    f"\nStored model accuracy against actual labels: "
    f"{100 * accuracy:.1f}%"
)

# Confusion table using actual lighting labels
real_confusion = pd.crosstab(
    pd.Series(y_test, name="Actual lighting"),
    pd.Series(predicted_names, name="Predicted lighting")
)

print("\nActual vs Predicted Lighting:")
print(real_confusion)

# Save results for the report
real_confusion.to_csv("../data/test_confusion.csv")
