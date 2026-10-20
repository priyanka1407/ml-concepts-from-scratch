"""
Linear Regression from scratch with Gradient Descent (NumPy only).

Story: predict house price (in lakhs) from house size (in 100 sq ft).
The model is a straight line:  price = w * size + b
We start with w = 0, b = 0 and nudge them step by step to reduce the error.
"""
import numpy as np

# 1. Toy data: the "true" rule is price = 3 * size + 5, plus some noise
rng = np.random.default_rng(42)
X = rng.uniform(5, 20, size=50)              # house sizes
y = 3 * X + 5 + rng.normal(0, 2, size=50)    # prices with noise

# 2. Start with a bad guess
w, b = 0.0, 0.0
learning_rate = 0.005
epochs = 2000

for epoch in range(epochs):
    # 3. Forward pass: predict with the current line
    y_pred = w * X + b

    # 4. Loss = Mean Squared Error (average of squared mistakes)
    error = y_pred - y
    loss = np.mean(error ** 2)

    # 5. Gradients: how much the loss changes if we nudge w or b
    dw = 2 * np.mean(error * X)
    db = 2 * np.mean(error)

    # 6. Update: step a little in the direction that lowers the loss
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 400 == 0:
        print(f"epoch {epoch:4d}  loss={loss:8.3f}  w={w:.3f}  b={b:.3f}")

print(f"\nLearned line: price = {w:.2f} * size + {b:.2f}   (true: 3 * size + 5)")

# 7. Compare with the closed-form solution (Normal Equation)
X_design = np.column_stack([X, np.ones_like(X)])
w_exact, b_exact = np.linalg.lstsq(X_design, y, rcond=None)[0]
print(f"Normal equation: price = {w_exact:.2f} * size + {b_exact:.2f}")

# 8. Use the model
print(f"Predicted price for size 12: {w * 12 + b:.1f} lakhs")
