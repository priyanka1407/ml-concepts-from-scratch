# 06 · Evaluation Metrics (Precision, Recall, F1)

**One line:** accuracy alone can lie on imbalanced data — precision and recall tell the real story.

## Confusion matrix (fraud detector)
|  | Predicted fraud | Predicted normal |
|---|---|---|
| **Actual fraud** | TP — caught it ✅ | FN — missed it ❌ |
| **Actual normal** | FP — false alarm ⚠️ | TN — correctly ignored ✅ |

## Formulas in plain words
- **Accuracy** = (TP+TN) / all → "how often am I right overall?"
- **Precision** = TP / (TP+FP) → "when I shout *fraud*, how often am I right?"
- **Recall** = TP / (TP+FN) → "of all real fraud, how much did I catch?"
- **F1** = harmonic mean of precision and recall → one balanced number.

## The trap
2% of transactions are fraud. A model that **always says "normal"** gets **98% accuracy** and catches **zero fraud**.

## Which one matters?
| Use case | Care most about | Why |
|---|---|---|
| Fraud / cancer screening | Recall | Missing a case is expensive |
| Spam filter | Precision | Hiding a real email is annoying |
| Balanced need | F1 | Trade-off between both |

Moving the decision **threshold** trades precision against recall.

## Run
```bash
python metrics.py
```

## Interview line
> "On imbalanced data I don't trust accuracy. I look at the confusion matrix and pick the metric that matches the business cost — recall when misses are expensive, precision when false alarms are — and tune the threshold to hit that trade-off."
