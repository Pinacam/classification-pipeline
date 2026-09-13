# Lab 03 — Recorded Results

## Part 1 — DBSCAN

### Baseline — scaled features, eps=0.06

```text
3 clusters; 33 noise
 cluster 0: 209 points
 cluster 1: 198 points
 cluster 2: 178 points
```

Figure:

```text
figures/generated/clustered_space.pdf
```

### k-means comparison

Even with k=3, k-means slices the crescents and ring because it assigns
points to the nearest centre. DBSCAN works better for these non-convex
shapes.

---

### eps=0.03 — scaled features

```text
15 clusters; 82 noise
```

Small eps fragments the shapes into many smaller clusters and creates
more noise.

---

### eps=0.30 — scaled features

```text
1 cluster; 0 noise
```

Large eps connects the shapes together and merges everything into one
cluster.

---

### Raw vs Scaled Features

| Features | eps=0.06 | eps=0.30 |
|---|---|---|
| Raw | 0 clusters; 618 noise | 10 clusters; 50 noise |
| Scaled [0,1]^2 | 3 clusters; 33 noise | 1 cluster; 0 noise |

Raw eps=0.30 cluster sizes:

```text
10 clusters; 50 noise
 cluster 0: 207 points
 cluster 1: 49 points
 cluster 2: 176 points
 cluster 3: 16 points
 cluster 4: 21 points
 cluster 5: 33 points
 cluster 6: 12 points
 cluster 7: 11 points
 cluster 8: 32 points
 cluster 9: 11 points
```

### Scaling conclusion

The same eps behaves very differently depending on feature scaling.

Scaled eps=0.06 finds the three shapes, while raw eps=0.06 sees all
618 training points as noise.

Scaled eps=0.30 merges everything, while raw eps=0.30 still fragments
the data.

This shows that eps must be reported together with the feature scaling.

---

## Part 2 — KDE Probability Maps

### Bandwidth Selection

Validation results:

| h | Mean log-likelihood |
|---|---:|
| 0.03 | 1.668 |
| 0.04 | 1.775 |
| 0.05 | 1.806 |
| 0.06 | 1.767 |
| 0.08 | 1.632 |
| 0.10 | 1.487 |
| 0.12 | 1.353 |

Selected bandwidth:

```text
h = 0.05
```

h=0.05 had the highest validation mean log-likelihood, so it was selected
without using the test set.

### Bandwidth sensitivity

At h=0.03, the KDE maps are sharper and more concentrated around the
data.

At h=0.12, the maps are smoother and more spread out. The ring's hollow
centre begins to fill in.

The validation-selected bandwidth h=0.05 gives the best result according
to the validation log-likelihood and was restored for the final model.

### KDE observations

At h=0.05, the continuous KDE maps preserve the two crescents and the
ring shape. The ring's hollow centre remains low-density.

DBSCAN noise points are not used when fitting the KDE maps. Including
them would spread density into areas that do not belong to a class and
distort the maps.

If the noise points represented real measurements instead of synthetic
scatter, removing all of them could also discard unusual but useful data.

The total distribution figure uses a fine display grid for visualization,
not the stored 16x16 grid.

Figures created:

```text
figures/generated/class_surfaces_continuous.pdf
figures/generated/total_distribution.pdf
```

Bandwidth comparison figures were also saved for h=0.03 and h=0.12.

---

## Part 3 — Model Storage

### 16x16 Grid

| Storage | Bytes/cell | Per class | All 3 classes |
|---|---:|---:|---:|
| uint8 | 1 | 256 B | 768 B |
| uint16 | 2 | 512 B | 1536 B |
| float32 | 4 | 1024 B | 3072 B |
| float64 | 8 | 2048 B | 6144 B |

Scaling bounds:

```text
16 bytes
```

uint8 worst-case error:

```text
0.195% of peak
```

For a 16x16 grid with 3 classes:

```text
3 x 16^2 x 1 byte = 768 bytes
```

I would use uint8 because the complete three-class model only needs
768 bytes plus 16 bytes for the scaling bounds, while the measured
quantization error is only 0.195% of peak.

This gives a large reduction in storage with very little error.

---

### 32x32 Grid Test

| Storage | Bytes/cell | Per class | All 3 classes |
|---|---:|---:|---:|
| uint8 | 1 | 1024 B | 3072 B |
| uint16 | 2 | 2048 B | 6144 B |
| float32 | 4 | 4096 B | 12288 B |
| float64 | 8 | 8192 B | 24576 B |

Scaling bounds:

```text
16 bytes
```

uint8 worst-case error:

```text
0.196% of peak
```

Increasing the grid from 16x16 to 32x32 improves the spatial resolution,
but storage increases by 4x because the footprint grows with G^2.

The 32x32 grid represents the shapes, including the ring's hollow centre,
with more spatial detail, but it costs much more memory.

The final configuration was restored to:

```text
GRID_N = 16
```

and the Part 2 and Part 3 files were rerun so the committed model uses
the 16x16 grid.

Figure:

```text
figures/generated/class_surfaces_discretized.pdf
```

---

## Part 4 — Classification

### Skeleton Classification Test

```text
[0.0, -1.8] -> 0
[0.0, 5.0] -> 2
[6.8, 1.6] -> 1
[4.8, 1.6] -> None
```

The first three points are classified into the three discovered classes.

The point [4.8, 1.6] is in the hollow centre of the ring and returns
None, meaning UNKNOWN.

---

### Held-Out Test Query

Query:

```text
[5.99958893 3.17868816]
```

Grid cell:

```text
row 11, col 13 of 16x16
```

### Nearest-cell lookup

```text
class 1
bytes = 0.0, 255.0, 0.0
distribution = 0.0%, 100.0%, 0.0%
```

### Bilinear lookup

```text
class 1
bytes = 0.0, 221.8, 0.0
distribution = 0.0%, 100.0%, 0.0%
```

Both methods choose class 1.

Nearest-cell lookup reads one byte per class, while bilinear lookup uses
four surrounding cells per class.

For this query, both methods give the same class and the same 100%
normalized class distribution.

---

### 3x3 Stored Byte Window

Class 0:

```text
   0    0    0
   0    0    0
   0    0    0
```

Class 1:

```text
 187  164   76
 208  255  183
  68  159  222
```

Class 2:

```text
   0    0    0
   0    0    0
   0    0    0
```

The class 1 values change strongly within only a few neighboring cells.
This shows that the stored decision surface can change sharply near grid
cell boundaries.

Bilinear lookup would have the biggest effect near these boundaries
because it blends the values of surrounding cells.

---

### Reject Region

```text
127 of 256 cells
49.6%
tau = 0.12
```

Almost half of the grid is below the rejection threshold.

The reject threshold is important because without it, the classifier
would be forced to guess a class even in low-density regions where no
class is a good match.

---

### Ring Centre Test

```text
ring centre [4.8 1.6] -> unknown
```

This is the correct result because the centre of the ring contains no
class data even though it is surrounded by the ring.

The classifier correctly rejects the point instead of incorrectly
assigning it to one of the classes.

Figure:

```text
figures/generated/decision_map.pdf
```
---

## Part 4 — Classification

### Skeleton Classification Test

```text
[0.0, -1.8] -> 0
[0.0, 5.0] -> 2
[6.8, 1.6] -> 1
[4.8, 1.6] -> None
```

The first three points were classified into the three discovered classes.

The point `[4.8, 1.6]` is in the hollow centre of the ring and returned
`None`, meaning UNKNOWN.

---

### Held-Out Test Query

Query:

```text
[5.99958893 3.17868816]
```

Grid cell:

```text
row 11, col 13 of 16x16
```

### Nearest-Cell Lookup

```text
class 1
bytes = 0.0, 255.0, 0.0
distribution = 0.0%, 100.0%, 0.0%
```

### Bilinear Lookup

```text
class 1
bytes = 0.0, 221.8, 0.0
distribution = 0.0%, 100.0%, 0.0%
```

Both methods chose class 1.

Nearest-cell lookup reads one byte per class, while bilinear lookup uses
four surrounding cells per class.

For this query, both methods gave the same class and the same 100%
normalized class distribution.

---

### 3x3 Stored Byte Window

Class 0:

```text
   0    0    0
   0    0    0
   0    0    0
```

Class 1:

```text
 187  164   76
 208  255  183
  68  159  222
```

Class 2:

```text
   0    0    0
   0    0    0
   0    0    0
```

The class 1 values change a lot within only a few neighboring cells.

This shows that the stored decision can change sharply near grid cell
boundaries.

Bilinear lookup can help near these boundaries because it blends the
values from surrounding cells.

---

### Reject Region

```text
127 of 256 cells
49.6%
tau = 0.12
```

Almost half of the grid is below the rejection threshold.

The reject threshold is important because without it, the classifier
would be forced to guess a class even in low-density regions where no
class is a good match.

---

### Ring Centre Test

```text
ring centre [4.8 1.6] -> unknown
```

This is the correct result because the centre of the ring does not
contain class data even though it is surrounded by the ring.

The classifier correctly rejects the point instead of assigning it to
one of the classes.

---

### Final Test Evaluation

Test points:

```text
206
```

Stored-vs-continuous agreement:

```text
100.0%
```

Confusion table:

```text
              c0    c1    c2   unk
   class 0    69     0     0     0
   class 1     0    68     0     0
   class 2     0     0    63     0
   unknown     0     0     0     6
```

The stored uint8 classifier agreed with the continuous KDE on all 206
held-out test points.

There were no disagreements between the stored model and the continuous
KDE.

The 69 class 0 points, 68 class 1 points, 63 class 2 points, and 6
unknown points were all reproduced correctly by the stored uint8 model.

This shows that the 16x16 uint8 model kept the same decisions as the
continuous KDE while using only 768 bytes for the three class tables
plus 16 bytes for the scaling bounds.

The bandwidth, grid size, and reject threshold were not changed after
using the test set.

Figure:

```text
figures/generated/decision_map.pdf
```