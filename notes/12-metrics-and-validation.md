# Metrics and validation

## Regression

- **MSE / RMSE**: penalizes large errors more than MAE.
- **R²**: fraction of variance explained. Rises when you add junk features; prefer adjusted R² or a holdout.

## Classification

- **Accuracy**: useless when one class dominates.
- **Precision**: TP / (TP + FP). Use when false alarms are expensive.
- **Recall**: TP / (TP + FN). Use when misses are expensive.
- **F1**: harmonic mean of precision and recall.
- **ROC-AUC**: can look fine on rare positives.
- **PR-AUC**: the plot that matters for fraud and churn.

## Top-k accuracy

An example is correct if fewer than `k` classes score strictly higher than its true class. Ties at the boundary count as correct.

## Confusion matrix

Rows = true class, columns = predicted class. Entry `(i, j)` is the count of true `i` predicted as `j`. The entries sum to the number of examples.

## Bias vs variance

- High bias: too simple, underfits.
- High variance: chases the sample, overfits.

## Cross-validation

- K-fold: every sample is tested exactly once. Stratify when a class is rare.
- Do not shuffle time series.
- Fit preprocessors inside the CV loop (or use a Pipeline) to avoid leakage.

## Hyperparameter search

- GridSearchCV: exhaustive, slow on large spaces.
- RandomizedSearchCV: samples a fixed number of combinations, usually good enough.
- Never tune on the final test set. Keep a completely untouched hold-out.

## Coding reference

Top-k accuracy and confusion matrix: [interview/coding_questions.md](../interview/coding_questions.md).
