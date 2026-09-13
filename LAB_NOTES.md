# Lab 03 — Recorded Results

## Part 1 — DBSCAN

### Baseline — Scaled Features, eps=0.06

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

The scaled data with `eps=0.06` found the three main shapes while
leaving 33 points as noise.

### k-means Comparison

Even with `k=3`, k-means does not represent the shapes as well because
it separates points based on distance from cluster centres.

DBSCAN works better for the crescents and ring because the shapes are
not simple circular clusters.

---

### eps=0.03 — Scaled Features

```text
15 clusters; 82 noise
```

Using a smaller `eps` caused the shapes to break into many smaller
clusters and increased the number of noise points.

---

### eps=0.30 — Scaled Features

```text
1 cluster; 0 noise
```

Using a larger `eps` connected the different shapes together and merged
everything into one cluster.

---

### Raw vs Scaled Features

| Features | eps=0.06 | eps=0.30 |
|---|---|---|
| Raw | 0 clusters; 618 noise | 10 clusters; 50 noise |
| Scaled [0,1]^2 | 3 clusters; 33 noise | 1 cluster; 0 noise |

Raw features with `eps=0.30`:

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

### Scaling Conclusion

The same `eps` value behaves very differently depending on whether the
features are scaled.

With scaled features, `eps=0.06` finds the three main shapes.

With raw features, the same `eps=0.06` treats all 618 training points
as noise.

With scaled features, `eps=0.30` merges everything into one cluster,
while the raw data still produces 10 clusters.

This shows that DBSCAN's `eps` value depends strongly on the scale of
the features.

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

`h=0.05` had the highest validation mean log-likelihood, so it was
selected for the final model.

The test set was not used when choosing the bandwidth.

---

### Bandwidth Sensitivity

At `h=0.03`, the KDE maps are sharper and more concentrated around the
data.

At `h=0.12`, the maps are smoother and more spread out.

With the larger bandwidth, the hollow centre of the ring begins to fill
in more.

The selected value `h=0.05` gives a balance between the two and had the
best validation score.

After the comparison, the bandwidth was restored to:

```text
h = 0.05
```

---

### KDE Observations

At `h=0.05`, the continuous KDE maps preserve the shape of the crescents
and the ring.

The hollow centre of the ring remains a low-density area.

DBSCAN noise points are excluded when the KDE models are fitted.

Including noise points could spread density into areas that do not
really belong to a class and could distort the probability maps.

If the noise points represented real unusual measurements, removing all
of them could also remove useful information.

The total distribution figure uses a fine grid for visualization rather
than the smaller grid used for the stored model.

Figures:

```text
figures/generated/class_surfaces_continuous.pdf
figures/generated/total_distribution.pdf
```

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

I would use `uint8` because the complete three-class model only needs
768 bytes plus 16 bytes for the scaling bounds.

The measured quantization error is only 0.195% of peak, so the storage
is much smaller while the error stays very low.

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
but it increases the storage by 4 times because the number of cells
depends on the square of the grid size.

The 32x32 grid gives more detail around the shapes and the hollow centre
of the ring, but it requires much more memory.

The final configuration was restored to:

```text
GRID_N = 16
```

The Part 2 and Part 3 scripts were rerun after restoring the 16x16 grid
so that the final stored model uses the correct grid size.

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

The first three test points were classified into the three discovered
classes.

The point `[4.8, 1.6]` is located in the hollow centre of the ring and
returned `None`, meaning UNKNOWN.

This shows that the reject threshold prevents the system from forcing a
class onto a point that does not belong strongly to any class.

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

---

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

Both lookup methods chose class 1.

Nearest-cell lookup reads one stored value per class.

With three classes, this means it reads 3 bytes.

Bilinear lookup uses four surrounding cells for each class.

With three classes, this means it reads 12 bytes.

For this test point, both methods produced the same final class and the
same normalized class distribution.

Nearest lookup is cheaper, while bilinear lookup can make transitions
between grid cells smoother.

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

The stored values for class 1 change a lot between nearby cells.

The centre cell has a value of 255, while some nearby cells have much
smaller values.

This shows that the stored probability can change sharply at grid cell
boundaries.

Bilinear lookup can reduce this effect because it blends values from the
surrounding cells.

---

### Reject Region

```text
127 of 256 cells
49.6%
tau = 0.12
```

Almost half of the grid is below the rejection threshold.

The reject threshold is important because without it, the classifier
would be forced to choose a class even in low-density areas where none
of the classes are a good match.

---

### Ring Centre Test

```text
ring centre [4.8 1.6] -> unknown
```

The ring centre was correctly classified as UNKNOWN.

This is the expected result because the centre of the ring is a
low-density region even though it is surrounded by points belonging to
the ring.

The classifier therefore avoids making an incorrect decision in this
area.

---

### Final Test Evaluation

Number of held-out test points:

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

There were no disagreements between the continuous KDE and the stored
uint8 lookup.

The results were:

```text
Class 0: 69 correct
Class 1: 68 correct
Class 2: 63 correct
Unknown: 6 correct
```

This gives:

```text
206 / 206 correct agreements
100.0% agreement
```

The result shows that the 16x16 uint8 lookup reproduces the decisions of
the continuous KDE on this held-out test set.

The stored model only requires:

```text
768 bytes for the three class tables
+ 16 bytes for scaling bounds
```

This makes the stored lookup much smaller and simpler for an embedded
device while keeping the same decisions as the continuous KDE on the
test set.

After the test set was used, the following parameters were kept frozen:

```text
BANDWIDTH = 0.05
GRID_N = 16
REJECT_TAU = 0.12
```

They were not changed after seeing the test results.

Figure:

```text
figures/generated/decision_map.pdf
```
