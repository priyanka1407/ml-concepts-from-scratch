# 02 · Logistic Regression (Binary Classification)

**One line:** linear regression + a sigmoid, so the output becomes a probability for a yes/no answer.

## Simple example
"Will this student pass?" Inputs: hours studied, hours slept.

```
score = w1·study + w2·sleep + b        (any number, e.g. -3 or 4)
prob  = sigmoid(score)                  (squashed to 0..1)
pass  = prob >= 0.5
```

| score | sigmoid(score) | decision |
|---|---|---|
| -4 | 0.02 | fail |
| 0 | 0.50 | borderline |
| 3 | 0.95 | pass |

## Why not MSE?
For probabilities we use **binary cross-entropy (log loss)**: it punishes *confident wrong* answers heavily. Predicting 0.99 for a student who failed costs a lot.

## Key terms
- **Sigmoid** `1 / (1 + e^-z)` — turns a score into a probability.
- **Decision boundary** — the line where prob = 0.5.
- **Feature scaling** — put features on similar ranges so gradient descent converges faster.

## Run
```bash
python logistic_regression.py
```

## Interview line
> "Logistic regression computes a linear score, passes it through a sigmoid to get a probability, and is trained with binary cross-entropy. Despite the name, it's a classification model, and its gradient is simply (prediction − label) × input."
