"""
Part 4 -- spend the TEST set once, on the frozen design.

There are no ground-truth labels because DBSCAN invented the classes,
so the honest question is whether the cheap stored uint8 model
reproduces the continuous KDE decision on unseen test data.
"""

import numpy as np

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
BANDWIDTH = 0.05


# Load data
data = load_split_scaled()

Xtr = data["train"]
Xte = data["test"]


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