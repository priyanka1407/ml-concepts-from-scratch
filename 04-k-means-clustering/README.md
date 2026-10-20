# 04 · K-Means Clustering (Unsupervised)

**One line:** group similar points together when you have **no labels**.

## Simple example — shop customers
We only know *spend* and *visits*. K-Means discovers groups like:
- Budget (low spend, few visits)
- Regular (medium spend, many visits)
- Premium (high spend)

## Algorithm (repeat until stable)
```
1. Pick k random centers
2. ASSIGN  : every point -> nearest center
3. UPDATE  : move every center -> average of its points
4. Repeat 2-3 until centers stop moving
```

## Choosing k — the elbow method
Plot *inertia* (total squared distance to centers) for k = 1, 2, 3 …
It always drops, but the drop flattens after the "right" k — the elbow.

## Gotchas
- Scale features first (distance-based).
- Random start can give different results → run several times (k-means++ helps).
- Assumes round-ish, similar-sized clusters.

## Run
```bash
python kmeans.py
```

## Interview line
> "K-Means alternates two steps — assign each point to its nearest centroid, then move each centroid to the mean of its points — until it converges. I choose k with the elbow method or silhouette score, and always scale features because it's distance-based."
