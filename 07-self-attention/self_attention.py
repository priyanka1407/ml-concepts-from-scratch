"""
Scaled dot-product self-attention from scratch (NumPy only) -
the core building block of Transformers and LLMs.

Sentence: "the cat sat on the mat"
Each word looks at every other word and decides how much to "pay attention" to it.
"""
import numpy as np

rng = np.random.default_rng(11)
tokens = ["the", "cat", "sat", "on", "the", "mat"]
seq_len, d_model, d_k = len(tokens), 8, 4

# 1. Pretend embeddings (in a real model these are learned)
X = rng.normal(0, 1, (seq_len, d_model))

# 2. Learned projection matrices turn each embedding into Query, Key, Value
W_q = rng.normal(0, 1, (d_model, d_k))
W_k = rng.normal(0, 1, (d_model, d_k))
W_v = rng.normal(0, 1, (d_model, d_k))

Q = X @ W_q   # "what am I looking for?"
K = X @ W_k   # "what do I contain?"
V = X @ W_v   # "what information do I pass on?"


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V, causal=False):
    # 3. Similarity of every query with every key, scaled by sqrt(d_k)
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    # 4. Causal mask (GPT-style): a word may not look at future words
    if causal:
        scores = np.where(np.triu(np.ones_like(scores), k=1) == 1, -np.inf, scores)
    # 5. Softmax turns scores into weights that sum to 1 per row
    weights = softmax(scores)
    # 6. Output = weighted mix of value vectors
    return weights @ V, weights


def show(weights, title):
    print(f"\n{title}")
    print("          " + " ".join(f"{t:>5}" for t in tokens))
    for t, row in zip(tokens, weights):
        print(f"{t:>8}  " + " ".join(f"{w:5.2f}" for w in row))


out, w = attention(Q, K, V)
show(w, "Encoder-style (BERT) attention weights - each row sums to 1:")

out_c, w_c = attention(Q, K, V, causal=True)
show(w_c, "Decoder-style (GPT) causal attention - upper triangle is 0:")

print(f"\nOutput shape: {out.shape}  (one new context-aware vector per token)")

# 7. Multi-head: run several attentions in parallel and concatenate
heads = []
for h in range(2):
    Wq, Wk, Wv = (rng.normal(0, 1, (d_model, d_k)) for _ in range(3))
    heads.append(attention(X @ Wq, X @ Wk, X @ Wv)[0])
multi = np.concatenate(heads, axis=-1)
print(f"Multi-head (2 heads) output shape: {multi.shape}")
