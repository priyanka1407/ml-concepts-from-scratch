# 03 · Decision Tree (Gini Impurity)

**One line:** a flowchart of yes/no questions learned from data.

## Simple example — loan approval
```
if credit_score <= 650:
    -> REJECT
else:
    if income <= 50:
        -> REJECT
    else:
        -> APPROVE
```
The tree picks *which question to ask first* by checking which one separates approved/rejected best.

## Gini impurity
`Gini = 1 − Σ p²` — how mixed a group is.

| Group | p(approve) | Gini |
|---|---|---|
| 10 approve, 0 reject | 1.0 | 0.0 (pure) |
| 5 approve, 5 reject | 0.5 | 0.5 (most mixed) |

At each node: try every feature × threshold, pick the split with the **biggest drop in Gini** (information gain).

## Stopping rules (prevent overfitting)
- `max_depth` — limit how many questions deep.
- `min_samples` — don't split tiny groups.
- Stop when a node is pure.

## Run
```bash
python decision_tree.py
```

## Interview line
> "A decision tree greedily chooses the feature and threshold that most reduces impurity — Gini or entropy — and recurses until a stopping rule. Trees are easy to explain but overfit, which is why Random Forests and gradient boosting combine many of them."
