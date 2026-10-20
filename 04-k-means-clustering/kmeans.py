"""
K-Means clustering from scratch (NumPy only).

Story: group shop customers by (annual spend, visits per month)
without any labels -> unsupervised learning.
"""
import numpy as np

rng = np.random.default_rng(7)

# 1. Three hidden customer groups
budget = rng.normal([20, 2], [4, 0.8], size=(40, 2))
regular = rng.normal([50, 8], [5, 1.0], size=(40, 2))
premium = rng.normal([90, 4], [6, 1.0], size=(40, 2))
X = np.vstack([budget, regular, premium])


def kmeans(X, k, iters=100, seed=0):
    r = np.random.default_rng(seed)
    # 2. Pick k random points as starting centroids
    centroids = X[r.choice(len(X), k, replace=False)]
    for i in range(iters):
        # 3. Assign: each point joins its nearest centroid
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        # 4. Update: move each centroid to the mean of its points
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        # 5. Stop when centroids stop moving
        if np.allclose(new_centroids, centroids):
            print(f"Converged after {i + 1} iterations")
            break
        centroids = new_centroids
    inertia = np.sum((X - centroids[labels]) ** 2)
    return centroids, labels, inertia


centroids, labels, _ = kmeans(X, k=3)
for j, c in enumerate(centroids):
    print(f"Cluster {j}: center spend={c[0]:.1f}k, visits={c[1]:.1f}/month, size={np.sum(labels == j)}")

# 6. Elbow method: how many clusters should we use?
print("\nElbow method (inertia = total squared distance to centers):")
for k in range(1, 7):
    _, _, inertia = kmeans(X, k)
    print(f"  k={k}: inertia={inertia:10.1f}")
print("The big drop stops around k=3 -> the 'elbow'.")
