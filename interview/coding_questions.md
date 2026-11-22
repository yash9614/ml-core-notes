# ML Coding Interview Questions

Numerically stable implementations of common ML primitives. All use only NumPy and the standard library (no SciPy / statsmodels). Solutions match the StrataScratch test cases.

## 1. Top-k Cosine Similarity Search

**Problem**: Given a query vector, a matrix of document embeddings (one row per document), and an integer `k`, return the indices of the `k` documents most similar to the query by cosine similarity, most similar first. Break ties by the lower index.

**Constraints**: No vector is all zeros. Return exactly `k` indices.

```python
import numpy as np

def top_k_cosine(query, docs, k):
    """
    Return indices of the k documents most similar to the query by cosine similarity.
    Ties broken by lower index.
    :rtype: List[int]
    """
    q = np.asarray(query, dtype=float)
    D = np.asarray(docs, dtype=float)
    sims = (D @ q) / (np.linalg.norm(D, axis=1) * np.linalg.norm(q))
    # Stable sort on negated similarities keeps original (lower) index order on ties
    return np.argsort(-sims, kind="stable")[:k].tolist()
```

## 2. Numerically Stable Softmax

**Problem**: Implement a row-wise softmax. Each output row must sum to 1. Must stay correct for very large logits (up to 1000 in absolute value).

```python
import numpy as np

def softmax(logits):
    """
    Row-wise numerically stable softmax.
    :rtype: List[List[float]]
    """
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)          # largest logit becomes 0
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).tolist()
```

## 3. Binary Cross-Entropy From Logits

**Problem**: Compute the mean binary cross-entropy loss from raw logits and binary labels. Must be correct and finite for any logits.

```python
import numpy as np

def bce_with_logits(logits, labels):
    """
    Mean binary cross-entropy from logits (numerically stable).
    :rtype: float
    """
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    # log(1 + exp(t)) without overflow
    loss = y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)
    return float(loss.mean())
```

## 4. Top-k Accuracy

**Problem**: Compute top-k accuracy. An example is correct if fewer than `k` classes score strictly higher than its true class (ties count in the true class's favor).

```python
import numpy as np

def top_k_accuracy(scores, labels, k):
    """
    Fraction of examples whose true class is among the k highest-scoring classes.
    :rtype: float
    """
    S = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    true_score = S[np.arange(len(y)), y][:, None]
    higher = (S > true_score).sum(axis=1)
    return float((higher < k).mean())
```

## 5. Gini Impurity and the Best Decision-Stump Split

**Problem**: Find the best single split for a decision stump using Gini impurity. For every feature, try every threshold halfway between two consecutive distinct values. Rows with value ≤ threshold go left. Score a split by the size-weighted average of the two children's Gini impurities. Return `[feature_index, threshold, weighted_gini]` for the lowest score. Break ties by smaller feature index, then smaller threshold.

```python
import numpy as np

def gini(labels):
    """Gini impurity of a set of labels."""
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - (p**2).sum()

def best_stump_split(X, y):
    """
    Best single split for a decision stump by size-weighted Gini.
    :rtype: List  # [feature_index, threshold, weighted_gini]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)
    best = None
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for t in (values[:-1] + values[1:]) / 2:
            left = X[:, j] <= t
            score = (
                left.sum() * gini(y[left])
                + (~left).sum() * gini(y[~left])
            ) / n
            # Prefer lower score; ties already broken by feature index then threshold order
            if best is None or score < best[2] - 1e-12:
                best = [j, float(t), float(score)]
    return best
```

## 6. LoRA Forward Pass

**Problem**: Implement the forward pass of a linear layer with a LoRA adapter. Frozen weight `W` (out × in). Adapter adds a low-rank update through `A` (r × in) and `B` (out × r), scaled by `alpha / r`:

$$ y = x W^\top + \frac{\alpha}{r} x A^\top B^\top $$

Compute it without forming the full out × in update matrix. No bias.

```python
import numpy as np

def lora_forward(x, W, A, B, alpha):
    """
    Forward pass of a linear layer with LoRA adapter.
    :rtype: List[List[float]]  # shape (batch, out_features)
    """
    x, W, A, B = (np.asarray(a, dtype=float) for a in (x, W, A, B))
    r = A.shape[0]
    base = x @ W.T
    update = (x @ A.T) @ B.T          # never materializes the full update matrix
    return (base + (alpha / r) * update).tolist()
```

## 7. Logistic Regression With Gradient Descent

**Problem**: Train a binary logistic regression model with full-batch gradient descent. Predict `sigmoid(X @ w + b)`. Minimize mean binary cross-entropy plus an L2 penalty `(l2 / 2) * ||w||^2` on the weights only (bias is not regularized). Start from `w = 0`, `b = 0`. The sigmoid must stay numerically stable for very large inputs. Return the final weights followed by the bias as one flat list.

```python
import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid via tanh."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def train_logreg(X, y, lr, epochs, l2):
    """
    Full-batch logistic regression with L2 on weights only.
    :rtype: List[float]  # [w_1, ..., w_num_features, b]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, num_features = X.shape
    w = np.zeros(num_features)
    b = 0.0
    for _ in range(epochs):
        err = sigmoid(X @ w + b) - y
        grad_w = X.T @ err / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w.tolist() + [float(b)]
```

## 8. Confusion Matrix

**Problem**: Build a confusion matrix. Entry `[i][j]` counts the examples with true class `i` that were predicted as class `j`. Rows are true classes, columns are predicted classes.

```python
import numpy as np

def confusion_matrix(y_true, y_pred, n_classes):
    """
    Confusion matrix. Rows = true class, columns = predicted class.
    :rtype: List[List[int]]
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm.tolist()
```

## 9. Cross-Entropy Loss From Scratch

**Problem**: Implement mean cross-entropy loss for a batch. Given raw logits of shape `(batch_size, num_classes)` and integer class labels of shape `(batch_size,)`, return the average loss. Must stay numerically stable for large logits.

```python
import numpy as np

def cross_entropy(logits, labels):
    """
    Mean cross-entropy from logits (numerically stable).
    :rtype: float
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())
```

---

**Sources**: StrataScratch ML coding questions (IDs 10713, 10751, 10746, 10754, 10745, 10740, 10722, 10755, 10707).

See also the conceptual interview bank in [questions.md](questions.md).
