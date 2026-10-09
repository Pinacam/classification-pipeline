"""Shared setup: the dataset, feature scaling, the lookup grid,
and byte storage. Defined once here so every stage agrees
on all four.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from pathlib import Path
import pandas as pd

SEED = 20250903

# Flash cost is classes * GRID_N**2 bytes, so this
# is a memory decision.
GRID_N = 16

# Below this, the winning class is too weak to
# trust: answer "unknown".
REJECT_TAU = 0.12

# Plot axis limits
VIEW_LIM = (-0.1, 1.1)

# Three-way split
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20


def _arc(rng, cx, cy, r, a0_deg, a1_deg, n, jitter):
    """n noisy points on the arc of a circle -- a crescent or a full ring."""
    a = np.linspace(np.radians(a0_deg), np.radians(a1_deg), n)

    pts = np.column_stack([
        cx + r * np.cos(a),
        cy + r * np.sin(a)
    ])

    return pts + rng.normal(0.0, jitter, pts.shape)

def load_dataset(csv_path=None):
    """Load real UV and IR sensor data from the CSV."""

    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "data_complete.csv"
        )

    data = pd.read_csv(csv_path)

    # Only use sensor measurements for classification
    X = data[["uv_sensor", "IR_sensor"]].to_numpy(dtype=float)

    return X

def load_labeled_dataset():
    """Load sensor readings and their real labels."""

    csv_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "data_complete.csv"
    )

    data = pd.read_csv(csv_path)

    X = data[["uv_sensor", "IR_sensor"]].to_numpy(dtype=float)
    y = data["label"].to_numpy()

    return X, y


def scale_features(X, bounds=None):
    """Apply logarithmic IR scaling, then min-max scale both sensors."""

    # Make a copy so the original readings stay unchanged
    X = np.asarray(X, dtype=float).copy()

    # Reduce the effect of very large IR readings
    X[:, 1] = np.log1p(X[:, 1])

    # Use training bounds for validation and testing too
    lo, hi = bounds if bounds is not None else (
        X.min(axis=0),
        X.max(axis=0)
    )

    # Scale both features between 0 and 1
    X_scaled = (X - lo) / (hi - lo)

    return X_scaled, (lo, hi)




def split_data(X, y, seed=SEED):
    """Split sensor data and labels into train, validation, and test."""

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y,
        train_size=TRAIN_FRAC,
        random_state=seed,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest,
        test_size=0.5,
        random_state=seed,
        stratify=y_rest
    )

    return X_train, X_val, X_test, y_train, y_val, y_test




def load_split_scaled(seed=SEED):
    """Load, split and scale the real sensor data."""

    # Load sensor measurements and their real labels
    X, y = load_labeled_dataset()

    # Split into training, validation and testing
    train, val, test, y_train, y_val, y_test = split_data(
        X, y, seed
    )

    # Scale using only the training data bounds
    train_s, bounds = scale_features(train)
    val_s, _ = scale_features(val, bounds)
    test_s, _ = scale_features(test, bounds)

    return {
        "train": train_s,
        "val": val_s,
        "test": test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "bounds": bounds
    }



def make_grid():
    """Cell centres of the GRID_N x GRID_N grid covering [0, 1]^2."""

    centres = (np.arange(GRID_N) + 0.5) / GRID_N
    xx, yy = np.meshgrid(centres, centres)

    return xx, yy, np.column_stack([
        xx.ravel(),
        yy.ravel()
    ])


def cell_of(point):
    """The (row, col) cell holding a scaled point."""

    i = np.clip(
        (np.asarray(point) * GRID_N).astype(int),
        0,
        GRID_N - 1
    )

    return int(i[1]), int(i[0])


def fit_density(points, bandwidth):
    """Fit a Gaussian KDE."""

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth
    )

    return kde.fit(points)


def evaluate_on_grid(kde, grid_points, xx):
    """Evaluate a fitted KDE at every cell centre."""

    return np.exp(
        kde.score_samples(grid_points)
    ).reshape(xx.shape)


def normalise_peak(surface):
    """Scale a density map so its own peak is 1.0."""

    peak = surface.max()

    return surface / peak if peak > 0 else surface


def quantize(surface_01):
    """Store a peak-normalised map as one byte per cell."""

    return np.round(
        np.clip(surface_01, 0.0, 1.0) * 255
    ).astype(np.uint8)


def dequantize(codes):
    """Read stored bytes back as values in [0, 1]."""

    return codes.astype(float) / 255.0