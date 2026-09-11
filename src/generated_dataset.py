import numpy as np
from pipeline_common import load_dataset

X = load_dataset()

np.savetxt(
    "dataset.csv",
    X,
    delimiter=",",
    fmt="%.6f"
)

print(
    f"wrote dataset.csv: {X.shape[0]} points, "
    f"{X.shape[1]} features"
)