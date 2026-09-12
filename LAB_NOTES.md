# Lab 03 — Recorded Results

## Part 1 — DBSCAN

### Baseline — scaled features, eps=0.06

```text
3 clusters; 33 noise
 cluster 0: 209 points
 cluster 1: 198 points
 cluster 2: 178 points
```

Figure: `figures/generated/clustered_space.pdf`

k-means observation:
Even with k=3, k-means slices the crescents and ring because it assigns points
to the nearest centre. DBSCAN works better for these non-convex shapes.

---

### eps=0.03 — scaled features

```text
15 clusters; 82 noise
```

Small eps fragments the shapes into many smaller clusters and creates more noise.

---

### eps=0.30 — scaled features

```text
1 cluster; 0 noise
```

Large eps connects the shapes together and merges everything into one cluster.

---

## Table 1 — Raw vs Scaled Features

| Features | eps=0.06 | eps=0.30 |
|---|---|---|
| Raw | 0 clusters; 618 noise | 10 clusters; 50 noise |
| Scaled `[0,1]^2` | 3 clusters; 33 noise | 1 cluster; 0 noise |

### Raw eps=0.30 cluster sizes

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

The same eps behaves differently depending on feature scale.

Scaled `eps=0.06` finds the three shapes, while raw `eps=0.06` sees all points
as noise.

Scaled `eps=0.30` merges everything, while raw `eps=0.30` still fragments the data.

This shows that eps must be reported together with the feature scaling.

---

## Final Part 1 Configuration

Restore before committing:

```python
X = load_split_scaled()["train"]
db = DBSCAN(eps=0.06, min_samples=8)
```

Expected final output:

```text
3 clusters; 33 noise
 cluster 0: 209 points
 cluster 1: 198 points
 cluster 2: 178 points
```

## Part 2 — KDE Bandwidth Selection

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

`h=0.05` had the highest validation mean log-likelihood, so it was selected without using the test set.

### Part 2 — KDE Observations

At the selected bandwidth `h=0.05`, the continuous KDE maps keep the two crescents and the ring shape. The ring's hollow centre remains low-density.

DBSCAN noise points are not included when fitting the class KDEs. Including them would spread density into areas that do not belong to a class and distort the maps. If those noise points were real measurements instead of synthetic scatter, completely discarding them could also throw away useful unusual data.

`total_distribution.pdf` uses a fine display grid for visualization, not the stored 16x16 grid.

At h=0.03, the KDE maps are sharper and more fragmented around the data points.

At h=0.12, the maps are much smoother and more spread out. The ring's hollow centre begins to fill in.

The validation-selected bandwidth h=0.05 gives the best balance and was restored for the final model.

## Part 3 — Model Storage

| Storage | Bytes/cell | Per class | All 3 classes |
|---|---:|---:|---:|
| uint8 | 1 | 256 B | 768 B |
| uint16 | 2 | 512 B | 1536 B |
| float32 | 4 | 1024 B | 3072 B |
| float64 | 8 | 2048 B | 6144 B |

Scaling bounds: 16 bytes

uint8 worst-case error: 0.195% of peak

For a 16x16 grid with 3 classes:

3 × 16² × 1 byte = 768 bytes

I would use uint8 because the complete three-class model only needs 768 bytes
plus 16 bytes for the scaling bounds, while the measured error is only 0.195%.
This gives a large storage reduction with very little loss in accuracy.

### 32x32 Grid Test

With GRID_N = 32:

| Storage | Bytes/cell | Per class | All 3 classes |
|---|---:|---:|---:|
| uint8 | 1 | 1024 B | 3072 B |
| uint16 | 2 | 2048 B | 6144 B |
| float32 | 4 | 4096 B | 12288 B |
| float64 | 8 | 8192 B | 24576 B |

Scaling bounds: 16 bytes

uint8 worst-case error: 0.196% of peak

Increasing the grid from 16x16 to 32x32 improves spatial resolution,
but storage increases by 4x because the footprint grows with G².

The 32x32 grid represents the ring's hollow centre more clearly,
but it costs much more memory.
