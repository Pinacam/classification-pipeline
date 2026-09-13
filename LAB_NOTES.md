# Lab 03 — Recorded Results

## Part 1 — DBSCAN

### Baseline — scaled features, eps=0.06

```text
3 clusters; 33 noise
 cluster 0: 209 points
 cluster 1: 198 points
 cluster 2: 178 points
```
Figure: figures/generated/clustered_space.pdf (Diagram B, DBSCAN vs k-means).
k-means observation: given the correct k=3, k-means still slices each crescent
across its middle and assigns the ring's hollow interior region by nearest
centre — its "nearest centre" rule can only cut the plane into straight-edged
(convex) pieces, so non-convex shapes are unavoidably split. Not a tuning
problem: no k fixes it.

### Experiment: eps=0.03 (scaled features)
```
15 clusters; 82 noise
 cluster 0: 207 points
 cluster 1: 49 points
 cluster 2: 176 points
 cluster 3: 16 points
 cluster 4: 14 points
 cluster 5: 14 points
 cluster 6: 13 points
 cluster 7: 8 points
 cluster 8: 10 points
 cluster 9: 8 points
 cluster 10: 8 points
 cluster 11: 12 points
 cluster 12: 13 points
 cluster 13: 8 points
 cluster 14: 7 points
```
=> Small eps FRAGMENTS: halving the neighbourhood radius breaks the
density-reachability chains wherever the shapes have small gaps, so the three
shapes shatter into many small clusters and noise rises 33 -> 82.

### Experiment: eps=0.30 (scaled features)
```
1 clusters; 0 noise
 cluster 0: 618 points
```
=> Large eps MERGES: the neighbourhood radius (~30% of the feature range) is
larger than the gaps between the three shapes, so density-reachability chains
connect across shapes and everything collapses into one cluster with no noise.

Summary of the eps experiment (scaled features, min_samples=8):
  eps=0.03 -> 15 clusters, 82 noise  (fragments)
  eps=0.06 ->  3 clusters, 33 noise  (baseline: finds the three shapes)
  eps=0.30 ->  1 cluster,   0 noise  (merges everything)

## Table 1 — same eps, raw vs scaled features (training set)
| Features            | eps=0.06 | eps=0.30 |
|---------------------|----------|----------|
| raw                 | (pending)| (pending)|
| scaled to [0,1]^2   | 3 clusters; 33 noise | 1 cluster; 0 noise |
