# ML Concepts from Scratch

Core machine-learning ideas built **from scratch with NumPy only**: no scikit-learn, no PyTorch. Each folder has:
- a `README.md` that explains it in simple words, with an example and an interview-ready line
- one runnable Python file with line-by-line comments

| # | Concept | What you learn |
|---|---|---|
| 01 | [Linear Regression](01-linear-regression) | Gradient descent, MSE loss |
| 02 | [Logistic Regression](02-logistic-regression) | Sigmoid, cross-entropy, classification |
| 03 | [Decision Tree](03-decision-tree) | Gini impurity, greedy splits, overfitting |
| 04 | [K-Means Clustering](04-k-means-clustering) | Unsupervised learning, elbow method |
| 05 | [Neural Network + Backprop](05-neural-network-backprop) | Hidden layers, chain rule, XOR |
| 06 | [Evaluation Metrics](06-evaluation-metrics) | Precision, recall, F1, imbalanced data |
| 07 | [Self-Attention](07-self-attention) | Q/K/V, causal mask, multi-head attention (Transformers) |

## Setup
```bash
pip install numpy
cd 01-linear-regression
python linear_regression.py
```

Related: [llm-agents-mini-projects](https://github.com/priyanka1407/llm-agents-mini-projects): single-agent, multi-agent and memory techniques with LLMs.
