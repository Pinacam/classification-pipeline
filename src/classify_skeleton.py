"""
Part 4 skeleton -- YOU write the lookup.

The tables, the grid, and the scaling are done.
Fill in the two TODOs, then check against classify_point.py.
"""

import numpy as np

from pipeline_common import (
    load_split_scaled,
    scale_features,
    quantize,
    dequantize,
    GRID_N,
    REJECT_TAU
)


# Training bounds ship with the model
bounds = load_split_scaled()["bounds"]

# Load the stored probability surfaces
surfaces = np.load("surfaces.npz")

# Get class numbers
classes = sorted(
    int(k[1:])
    for k in surfaces.files
)


# The shipped model:
# one uint8 table for each class
tables = {
    c: quantize(surfaces[f"c{c}"])
    for c in classes
}


def classify(raw_point):
    """
    Return the class label for a RAW point,
    or None if the point is unknown.
    """

    # Scale raw point into [0, 1]
    point, _ = scale_features(
        np.asarray(raw_point),
        bounds
    )

    # TODO 1:
    # Convert scaled point into a grid cell
    col = int(point[0] * GRID_N)
    row = int(point[1] * GRID_N)

    # Keep indices inside valid grid range
    col = int(np.clip(col, 0, GRID_N - 1))
    row = int(np.clip(row, 0, GRID_N - 1))


    # TODO 2:
    # Read one stored byte from each class table
    # and convert it back to a density value
    values = np.array([
        dequantize(tables[c][row, col])
        for c in classes
    ])

    # Find class with highest value
    best = int(np.argmax(values))

    # Reject point if even the strongest class
    # is below the threshold
    if values[best] < REJECT_TAU:
        return None

    return classes[best]


if __name__ == "__main__":

    # One point on each shape,
    # then the hollow centre of the ring
    for q in (
        [0.0, -1.8],
        [0.0, 5.0],
        [6.8, 1.6],
        [4.8, 1.6]
    ):

        print(
            q,
            "->",
            classify(q)
        )