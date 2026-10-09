
"""
Part 3 -- generate model.h for ESP32 classification.

Converts the trained KDE surfaces into uint8 lookup tables.
Also saves the scaling bounds, class names, and reject threshold.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline_common import (
    load_split_scaled,
    quantize,
    GRID_N,
    REJECT_TAU
)


# Project folders
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUT = PROJECT_DIR / "firmware" / "model.h"


# Load the trained KDE surfaces
surfaces = np.load(SRC_DIR / "surfaces.npz")

# Get the discovered class numbers
classes = sorted(
    int(k[1:])
    for k in surfaces.files
)

K = len(classes)


# Load the original training data and scaling bounds
data = load_split_scaled()

lo, hi = data["bounds"]

# Bounds are stored in this order:
# UV minimum, UV maximum, log(IR) minimum, log(IR) maximum
bounds = [
    lo[0],
    hi[0],
    lo[1],
    hi[1]
]


# Match DBSCAN clusters to actual lighting conditions
labels = np.loadtxt(
    SRC_DIR / "labels.csv",
    dtype=int
)

confusion = pd.crosstab(
    pd.Series(data["y_train"], name="Actual"),
    pd.Series(labels, name="Cluster")
)

class_names = [
    confusion[c].idxmax()
    for c in classes
]


# Convert KDE surfaces into uint8 tables
tables = [
    quantize(surfaces[f"c{c}"])
    for c in classes
]


# Make sure each table matches the grid size
for table in tables:
    assert table.shape == (GRID_N, GRID_N)


# Create the C header file
lines = [
    "// Generated KDE lookup model for ESP32",
    "// IR readings must use log1p() before scaling",
    "",
    "#ifndef MODEL_H",
    "#define MODEL_H",
    "",
    "#include <stdint.h>",
    "",
    "#ifndef PROGMEM",
    "#define PROGMEM",
    "#endif",
    "",
    f"#define GRID_N {GRID_N}",
    f"#define NUM_CLASSES {K}",
    f"#define REJECT_TAU {REJECT_TAU}f",
    "",
]


# Write scaling bounds
lines.append("// UV min, UV max, log(IR) min, log(IR) max")

bound_values = ", ".join(
    f"{float(v):.9f}f"
    for v in bounds
)

lines.append(
    f"const float SCALE_BOUNDS[4] = {{{bound_values}}};"
)

lines.append("")


# Write class names
names = ", ".join(
    f'"{name}"'
    for name in class_names
)

lines.append(
    f"const char* CLASS_NAMES[NUM_CLASSES] = {{{names}}};"
)

lines.append("")


# Write uint8 lookup table
lines.append(
    "const uint8_t CLASS_TABLE"
    "[NUM_CLASSES][GRID_N][GRID_N] PROGMEM = {"
)

for i, table in enumerate(tables):

    lines.append(
        f"    // Class {i}: {class_names[i]}"
    )

    lines.append("    {")

    for row in table:
        values = ", ".join(
            str(int(value))
            for value in row
        )

        lines.append(f"        {{{values}}},")

    lines.append("    },")

lines.append("};")

lines.append("")
lines.append("#endif // MODEL_H")


# Save model.h into the firmware folder
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)


# Print model information
table_bytes = K * GRID_N * GRID_N

print(f"Generated: {OUTPUT}")
print(f"Classes: {class_names}")
print(f"Grid size: {GRID_N} x {GRID_N}")
print(f"Lookup table: {table_bytes} bytes")
print(f"Scaling bounds: 16 bytes")
print(f"Table + bounds: {table_bytes + 16} bytes")
