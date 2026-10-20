"""
Decision Tree classifier from scratch using Gini impurity (pure Python + NumPy).

Story: should we approve a loan? Features: income (k/month), credit score.
"""
import numpy as np


def gini(labels):
    """Gini impurity: 0 = pure group, 0.5 = 50/50 mix (for 2 classes)."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)


def best_split(X, y):
    """Try every feature and threshold; keep the split with the lowest weighted Gini."""
    best = {"gain": 0.0}
    parent = gini(y)
    for feature in range(X.shape[1]):
        for threshold in np.unique(X[:, feature]):
            left = X[:, feature] <= threshold
            right = ~left
            if left.all() or right.all():
                continue
            weighted = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / len(y)
            gain = parent - weighted
            if gain > best["gain"]:
                best = {"gain": gain, "feature": feature, "threshold": threshold}
    return best


def build_tree(X, y, depth=0, max_depth=3, min_samples=2):
    """Recursively split until the node is pure, too small, or too deep."""
    majority = int(np.bincount(y).argmax())
    if depth >= max_depth or len(y) < min_samples or gini(y) == 0:
        return {"leaf": majority}
    split = best_split(X, y)
    if split["gain"] == 0:
        return {"leaf": majority}
    mask = X[:, split["feature"]] <= split["threshold"]
    return {
        "feature": split["feature"],
        "threshold": split["threshold"],
        "left": build_tree(X[mask], y[mask], depth + 1, max_depth, min_samples),
        "right": build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_samples),
    }


def predict_one(tree, x):
    while "leaf" not in tree:
        tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["leaf"]


def show(tree, names, indent=""):
    if "leaf" in tree:
        print(indent + ("-> APPROVE" if tree["leaf"] else "-> REJECT"))
        return
    print(f"{indent}if {names[tree['feature']]} <= {tree['threshold']:.0f}:")
    show(tree["left"], names, indent + "    ")
    print(f"{indent}else:")
    show(tree["right"], names, indent + "    ")


# Toy data: approve when income is decent AND credit score is good
rng = np.random.default_rng(1)
income = rng.integers(20, 150, 120)
credit = rng.integers(300, 850, 120)
X = np.column_stack([income, credit])
y = ((income > 50) & (credit > 650)).astype(int)

tree = build_tree(X, y, max_depth=3)
print("Learned tree:")
show(tree, ["income", "credit_score"])

preds = np.array([predict_one(tree, x) for x in X])
print(f"\nTraining accuracy: {np.mean(preds == y):.2%}")
print("Applicant (income 80k, credit 720):", "APPROVE" if predict_one(tree, [80, 720]) else "REJECT")
print("Applicant (income 80k, credit 500):", "APPROVE" if predict_one(tree, [80, 500]) else "REJECT")
