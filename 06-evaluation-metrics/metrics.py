"""
Classification metrics from scratch: confusion matrix, accuracy,
precision, recall, F1 - and why accuracy can lie.

Story: a fraud detector. 1 = fraud, 0 = normal transaction.
"""
import numpy as np


def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))  # caught fraud
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))  # false alarm
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))  # missed fraud
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))  # correctly ignored
    return tp, fp, fn, tn


def report(name, y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0   # of flagged, how many were fraud?
    recall = tp / (tp + fn) if tp + fn else 0.0      # of all fraud, how many did we catch?
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\n== {name} ==")
    print(f"              predicted fraud  predicted normal")
    print(f"actual fraud       TP={tp:<4}          FN={fn:<4}")
    print(f"actual normal      FP={fp:<4}          TN={tn:<4}")
    print(f"accuracy={accuracy:.2%}  precision={precision:.2%}  recall={recall:.2%}  F1={f1:.2%}")


rng = np.random.default_rng(5)
# 1000 transactions, only 2% fraud -> imbalanced data
y_true = (rng.random(1000) < 0.02).astype(int)

# Model A: lazy model that always says "normal"
lazy = np.zeros_like(y_true)
report("Lazy model (always 'normal')", y_true, lazy)

# Model B: a real-ish model that catches most fraud but raises some false alarms
real = y_true.copy()
miss = (y_true == 1) & (rng.random(1000) < 0.2)    # misses 20% of fraud
alarm = (y_true == 0) & (rng.random(1000) < 0.03)  # 3% false alarms
real[miss] = 0
real[alarm] = 1
report("Real model", y_true, real)

print("\nLesson: the lazy model has ~98% accuracy but 0% recall - useless for fraud.")

# Threshold trade-off with probabilities
scores = np.clip(y_true * 0.6 + rng.normal(0.2, 0.15, 1000), 0, 1)
print("\nThreshold trade-off (higher threshold -> more precise, catches less):")
for t in [0.3, 0.5, 0.7]:
    pred = (scores >= t).astype(int)
    tp, fp, fn, _ = confusion(y_true, pred)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    print(f"  threshold={t}: precision={p:.2%} recall={r:.2%}")
