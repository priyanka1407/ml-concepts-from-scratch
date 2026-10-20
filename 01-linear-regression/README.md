# 01 · Linear Regression (Gradient Descent)

**One line:** fit the best straight line through data so you can predict a number.

## Simple example
| House size (100 sq ft) | Price (lakhs) |
|---|---|
| 10 | 35 |
| 15 | 50 |
| 20 | 65 |

The rule hiding here is `price = 3 × size + 5`. Linear regression finds `3` (weight **w**) and `5` (bias **b**) by itself.

## How it learns (gradient descent)
1. Start with a bad guess: `w = 0, b = 0`.
2. Predict: `y_pred = w·x + b`.
3. Measure the mistake with **MSE** = average of `(y_pred − y)²`.
4. Compute the **gradient** — which direction makes the mistake smaller.
5. Take a small step that way (`learning_rate`). Repeat.

```
loss
 |\
 | \
 |  \___        <- each step walks downhill
 |      \____•  <- minimum = best w, b
 +------------------ w
```

## Key terms
- **Learning rate** — step size. Too big → jumps around; too small → very slow.
- **Epoch** — one pass over all data.
- **Normal equation** — exact formula answer; fine for small data, gradient descent scales better.

## Run
```bash
python linear_regression.py
```

## Interview line
> "Linear regression models the target as a weighted sum of features. I minimise mean squared error with gradient descent: compute the gradient of the loss w.r.t. weights and bias, step against it by a learning rate, and repeat until the loss stops falling."
