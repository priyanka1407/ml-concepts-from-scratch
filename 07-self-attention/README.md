# 07 · Self-Attention (the heart of Transformers & LLMs)

**One line:** every word looks at every other word and builds a new meaning from the ones that matter.

## Simple example
"The **cat** sat on the mat because **it** was tired."
To understand **it**, the model should attend strongly to **cat**. Attention learns those links.

## Query, Key, Value — library analogy
- **Query (Q)** — the question you bring: "what am I looking for?"
- **Key (K)** — the label on each book: "what do I contain?"
- **Value (V)** — the book content you actually take away.

```
scores  = Q · Kᵀ / √d_k      (how well each query matches each key)
weights = softmax(scores)    (turn into percentages per row)
output  = weights · V        (blend the values)
```
`√d_k` keeps scores from getting huge so softmax doesn't become one-hot too early.

## Encoder vs decoder
| | Sees | Example |
|---|---|---|
| Encoder (bidirectional) | all words | BERT — understanding/classification |
| Decoder (causal mask) | only past words | GPT/Claude — text generation |

## Multi-head attention
Run several attentions in parallel — one head may track grammar, another meaning — then concatenate.

## Run
```bash
python self_attention.py
```

## Interview line
> "Self-attention projects each token into query, key and value vectors, scores every query against every key with a scaled dot product, softmaxes those scores into weights, and outputs a weighted sum of values. Decoder LLMs add a causal mask so tokens can't see the future, and multi-head attention lets the model learn several relationship types at once."
