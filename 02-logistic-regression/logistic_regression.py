"""
Logistic Regression from scratch (NumPy only).

Story: predict whether a student passes (1) or fails (0)
from hours studied and hours slept.
"""
import numpy as np


def sigmoid(z):
    """Squash any number into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))


# 1. Toy data: more study + more sleep -> more likely to pass
rng = np.random.default_rng(0)
n = 200
study = rng.uniform(0, 10, n)
sleep = rng.uniform(3, 9, n)
X = np.column_stack([study, sleep])
true_score = 1.2 * study + 0.8 * sleep - 11
y = (true_score + rng.normal(0, 1, n) > 0).astype(int)

# 2. Scale features so gradient descent behaves nicely
mean, std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - mean) / std

# 3. Parameters
w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1

for epoch in range(1000):
    # Forward pass: linear score -> probability
    p = sigmoid(X_scaled @ w + b)

    # Binary cross-entropy loss (small epsilon avoids log(0))
    eps = 1e-9
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    # Gradients (neat result: just (p - y) times the input)
    dw = X_scaled.T @ (p - y) / n
    db = np.mean(p - y)

    w -= lr * dw
    b -= lr * db

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss={loss:.4f}")

# 4. Evaluate
pred = (sigmoid(X_scaled @ w + b) >= 0.5).astype(int)
print(f"\nTraining accuracy: {np.mean(pred == y):.2%}")

# 5. Predict a new student: 7 hours study, 8 hours sleep
new = (np.array([7, 8]) - mean) / std
prob = sigmoid(new @ w + b)
print(f"Student (7h study, 8h sleep) pass probability: {prob:.2%}")
