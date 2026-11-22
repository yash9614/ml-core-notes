# Concepts from Coding Questions

Key ideas that appear when you implement the primitives in [interview/coding_questions.md](../interview/coding_questions.md).

## Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring magnitude:

```math
\cos(\theta) = \frac{q \cdot d}{\|q\| \, \|d\|}
```

Two documents that point in the same direction get the same score even if one is longer. Useful for embeddings in retrieval (RAG) because length is often an artifact of how the vector was produced, not of meaning.

**Tie-breaking**: when similarities are equal, a stable sort on the negated scores preserves the original (lower) index order.

## Numerically Stable Softmax

Softmax is invariant to adding a constant to every logit in a row:

```math
\mathrm{softmax}(z) = \mathrm{softmax}(z - c)
```

Choose $c = \max z$ so the largest logit becomes 0. Then every exponent is ≤ 1 and `exp` cannot overflow. A naive `exp(z)` fails when logits reach ~700+.

## Log-Softmax and Cross-Entropy

Instead of computing softmax then taking `log`, compute the log-probabilities directly:

```math
\log \mathrm{softmax}(z)_k = z_k - \log\sum_j e^{z_j}
```

After the same max-shift, this stays finite. Cross-entropy is just the negative of the log-probability of the true class, averaged over the batch.

## Binary Cross-Entropy From Logits

The textbook form `log(sigmoid(z))` breaks when the sigmoid saturates to 0 or 1. Rewrite in terms of the logit:

```math
\begin{aligned}
\ell(y=1, z) &= \log(1 + e^{-z}) \\
\ell(y=0, z) &= \log(1 + e^{z})
\end{aligned}
```

`np.logaddexp(0, t)` computes `log(1 + exp(t))` without overflow. The two terms are switched by the label.

## Stable Sigmoid

The expression `1 / (1 + exp(-z))` overflows for large positive or negative `z`. An equivalent form that stays finite is:

```math
\sigma(z) = \frac{1}{2}\bigl(1 + \tanh(z/2)\bigr)
```

## Top-k Accuracy

An example is correct under top-k if fewer than `k` classes score strictly higher than the true class. Ties at the boundary count as correct (the true class is not beaten by `k` others). This is cheaper than sorting every row.

## Gini Impurity and Decision Stumps

Gini impurity of a set of labels:

```math
G = 1 - \sum_k p_k^2
```

A split is scored by the size-weighted average of the two children:

```math
\text{score} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R
```

Thresholds are placed halfway between consecutive distinct feature values. Ties are broken by smaller feature index, then smaller threshold.

## LoRA (Low-Rank Adaptation)

A frozen weight matrix `W` is adapted by a low-rank update:

```math
y = x W^\top + \frac{\alpha}{r} x A^\top B^\top
```

`A` is `r × in`, `B` is `out × r`. Computing `(x @ A.T) @ B.T` never materializes the full `out × in` update, so memory stays proportional to the rank. When `B` is zero (common initialization), the output equals the frozen layer.

## Logistic Regression Gradient

With mean binary cross-entropy, the gradient of the loss with respect to the linear scores is simply `prediction - label`. From there:

```math
\begin{aligned}
\nabla_w J &= \frac{1}{n} X^\top (\hat{p} - y) + \lambda w \\
\nabla_b J &= \mathrm{mean}(\hat{p} - y)
\end{aligned}
```

The L2 penalty is applied only to the weights; the bias is never regularized.

## Confusion Matrix

Entry `(i, j)` counts examples whose true class is `i` and predicted class is `j`. Rows are true classes, columns are predicted classes. `np.add.at` accumulates repeated pairs instead of overwriting them.

---

Implementations: [interview/coding_questions.md](../interview/coding_questions.md).
