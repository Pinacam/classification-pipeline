"""
Part 2a -- one KDE map per class, on the lookup grid,
peak-normalised.

Saves surfaces.npz, which every later stage reads.
"""

import matplotlib.pyplot as plt
import numpy as np

from pipeline_common import (
    load_split_scaled,
    make_grid,
    fit_density,
    evaluate_on_grid,
    normalise_peak,
    VIEW_LIM
)


BANDWIDTH = 0.05

# Fine grid used only for the figure, not for storage
DISPLAY_N = 200


# KDE is fit on TRAINING data only
X = load_split_scaled()["train"]

# Labels produced by Part 1
labels = np.loadtxt("labels.csv", dtype=int)

# Ignore DBSCAN noise (-1)
classes = sorted(set(labels) - {-1})


# Shared lookup grid
xx, yy, grid_points = make_grid()


# Fit one KDE per class
kdes = {
    c: fit_density(
        X[labels == c],
        BANDWIDTH
    )
    for c in classes
}


# Evaluate each KDE on the lookup grid
# and normalise each class to its own peak
surfaces = {
    c: normalise_peak(
        evaluate_on_grid(
            kdes[c],
            grid_points,
            xx
        )
    )
    for c in classes
}


# Save the maps for Parts 3 and 4
np.savez(
    "surfaces.npz",
    **{
        f"c{c}": surface
        for c, surface in surfaces.items()
    }
)

# Continuous figure

# Fine display grid
g = np.linspace(0, 1, DISPLAY_N)

fxx, fyy = np.meshgrid(g, g)

fine = np.column_stack(
    [
        fxx.ravel(),
        fyy.ravel()
    ]
)


# Create one panel per discovered class
fig, axes = plt.subplots(
    1,
    len(classes),
    figsize=(3.0 * len(classes), 3.0),
    sharex=True,
    sharey=True
)


for ax, c in zip(np.atleast_1d(axes), classes):

    smooth = normalise_peak(
        np.exp(
            kdes[c].score_samples(fine)
        ).reshape(fxx.shape)
    )

    im = ax.imshow(
        smooth,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="Greys",
        interpolation="bilinear",
        aspect="equal",
        vmin=0,
        vmax=1
    )

    ax.scatter(
        X[labels == c][:, 0],
        X[labels == c][:, 1],
        s=2,
        c="white",
        edgecolors="black",
        linewidths=0.2,
        alpha=0.7
    )

    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)

    ax.set_title(
        f"class {c}, continuous",
        fontsize=9
    )

    ax.set_xlabel("scaled feature $x_1$")


np.atleast_1d(axes)[0].set_ylabel(
    "scaled feature $x_2$"
)


fig.colorbar(
    im,
    ax=axes,
    shrink=0.85,
    label="normalised density"
)


fig.savefig(
    "../figures/generated/class_surfaces_continuous.pdf",
    bbox_inches="tight",
    transparent=True
)