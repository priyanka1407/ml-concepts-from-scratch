# 05 · Neural Network & Backpropagation (XOR)

**One line:** layers of simple neurons; backprop uses the chain rule to tell every weight how much it caused the error.

## Why XOR?
| A | B | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single straight line separates the 1s from the 0s → logistic regression fails. A **hidden layer** bends the space so it becomes separable.

## The loop
```
FORWARD : input -> hidden (W1, sigmoid) -> output (W2, sigmoid) -> loss
BACKWARD: loss -> dW2 -> pass blame back -> dW1      (chain rule)
UPDATE  : W -= learning_rate * dW
```

## Neuron = linear + activation
`output = activation(w·x + b)` — without the activation, stacking layers is still just one big linear function.

## Key terms
- **Activation** (sigmoid, ReLU, tanh) — adds non-linearity.
- **Backpropagation** — efficient gradient calculation layer by layer, output → input.
- **Vanishing gradient** — sigmoid derivatives are ≤ 0.25, so deep sigmoid nets learn slowly; ReLU fixes much of this.

## Run
```bash
python neural_network.py
```

## Interview line
> "A neural network is stacked linear layers with non-linear activations. Training runs a forward pass to get the loss, then backpropagation applies the chain rule from the output back to every weight, and gradient descent updates them. XOR is the classic proof that you need a hidden layer."
