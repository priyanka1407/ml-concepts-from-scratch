"""
A tiny neural network from scratch that learns XOR (NumPy only).

XOR cannot be solved by one straight line, so we need a hidden layer.
Architecture: 2 inputs -> 4 hidden neurons (sigmoid) -> 1 output (sigmoid)
"""
import numpy as np

# 1. XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 2. Random small weights
rng = np.random.default_rng(3)
W1 = rng.normal(0, 1, (2, 4))
b1 = np.zeros((1, 4))
W2 = rng.normal(0, 1, (4, 1))
b2 = np.zeros((1, 1))
lr = 1.0

for epoch in range(5001):
    # 3. FORWARD pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)          # hidden layer output
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)          # final prediction

    loss = np.mean((a2 - y) ** 2)

    # 4. BACKWARD pass (chain rule, from output back to input)
    d_a2 = 2 * (a2 - y) / len(X)          # dLoss/da2
    d_z2 = d_a2 * a2 * (1 - a2)           # sigmoid'(z) = s * (1 - s)
    d_W2 = a1.T @ d_z2
    d_b2 = d_z2.sum(axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T                    # blame passed back to hidden layer
    d_z1 = d_a1 * a1 * (1 - a1)
    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0, keepdims=True)

    # 5. UPDATE weights
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 1000 == 0:
        print(f"epoch {epoch:4d}  loss={loss:.5f}")

print("\nPredictions after training:")
for inp, out in zip(X, a2):
    print(f"  {inp.astype(int)} -> {out[0]:.3f}  (rounded {int(out[0] > 0.5)})")
