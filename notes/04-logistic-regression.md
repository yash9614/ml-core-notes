# Logistic regression

## What it is

A linear model that outputs a probability of the positive class. The linear score is passed through a sigmoid so the prediction stays in (0, 1). The decision boundary is still a hyperplane.

## Sigmoid

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

**Numerically stable form** (avoids overflow for large |z|):

```math
\sigma(z) = \frac{1}{2}\bigl(1 + \tanh(z/2)\bigr)
```

## Loss

Binary cross-entropy (log loss). Do not use squared error on the probability.

**From logits** (stable, no intermediate sigmoid):

```math
\begin{aligned}
\ell(y=1, z) &= \log(1 + e^{-z}) \\
\ell(y=0, z) &= \log(1 + e^{z})
\end{aligned}
```

`np.logaddexp(0, t)` implements `log(1 + exp(t))` safely.

## Coefficients

A coefficient is a change in log-odds. `exp(w_j)` is an odds ratio.

## Multiclass

- One-vs-rest: one binary model per class.
- Softmax: one joint model. Probabilities sum to 1.

**Stable softmax**: subtract the row max before exponentiating.

## Training with gradient descent

Full-batch update (mean loss, L2 on weights only):

```math
\begin{aligned}
\nabla_w J &= \frac{1}{n} X^\top (\hat{p} - y) + \lambda w \\
\nabla_b J &= \mathrm{mean}(\hat{p} - y)
\end{aligned}
```

Start from zero weights and bias. The bias is never regularized.

## Threshold

0.5 is not sacred. Tune it on a precision-recall curve when classes are imbalanced or error costs differ.

## sklearn

Default penalty is L2. `C` is the inverse regularization strength. Use a pipeline so scaling is fit only on train.

## Coding reference

Stable sigmoid, BCE-from-logits, and gradient-descent training: [interview/coding_questions.md](../interview/coding_questions.md).
