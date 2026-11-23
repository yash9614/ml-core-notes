# Overfitting and Underfitting

Source: Krish Naik materials + practical intuition.

## Intuition

- **Underfitting (high bias)**: Model is too simple. Both training and validation error are high. The model cannot capture the underlying pattern.
- **Overfitting (high variance)**: Model is too complex. Training error is very low, but validation/test error is high. The model has memorized noise in the training data.

A good model sits in the middle: low enough training error and low validation error.

## Visual diagnosis

- Plot learning curves (training vs validation error as dataset size grows).
- Underfitting: both curves high and close.
- Overfitting: training error low, validation error much higher and possibly rising.
- Good fit: both curves low and converging.

## Common causes and fixes

| Problem | Typical cause | Fixes |
|---------|---------------|-------|
| Underfitting | Too few features, model too simple, high regularization | Add features / polynomial terms, reduce regularization, use a more complex model |
| Overfitting | Too many features, deep trees, too little data, low regularization | Regularization, simpler model, more data, early stopping, pruning, ensembles, dropout (neural nets) |

## Industrial relevance

- Always look at both training and validation metrics. A model with 99% training accuracy and 70% validation accuracy is overfitting.
- In production, prefer models that generalize. Use cross-validation and a held-out test set.
- Tree ensembles (Random Forest, Gradient Boosting) and regularization are the most common practical tools against overfitting on tabular data.
- Monitor for data drift after deployment — a model that was well-calibrated can start overfitting the old distribution.

## Interview questions

1. Explain overfitting and underfitting. How do you diagnose each?
2. What is the bias-variance tradeoff?
3. How do you prevent a decision tree from overfitting?
4. Why does adding more training data usually help with overfitting but not underfitting?
5. What role does regularization play in controlling overfitting?
6. How do learning curves help you decide whether to collect more data or change the model?
