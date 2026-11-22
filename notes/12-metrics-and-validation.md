# Metrics and validation

Source: `5.3-Performance+Metrics.pdf`, `7-Performance+Metrics.pdf`, `Types+Of+Cross+Validation.pdf`.

## Regression

- **MSE / RMSE**: penalizes large errors more than MAE. RMSE is in the same units as the target.
- **MAE**: linear penalty. More robust to outliers than RMSE.
- **R²**: fraction of variance explained. Rises when you add junk features; prefer adjusted R² or a holdout.
- **Adjusted R²**: penalizes the number of features. Useful when comparing models with different numbers of predictors.

## Classification

- **Accuracy**: useless when one class dominates.
- **Precision**: TP / (TP + FP). Use when false alarms are expensive.
- **Recall**: TP / (TP + FN). Use when misses are expensive.
- **F1**: harmonic mean of precision and recall. Balances the two.
- **ROC-AUC**: ranking quality across thresholds. Can look fine on rare positives.
- **PR-AUC**: the plot that matters for fraud and churn. Focuses on the positive class.
- **Log loss**: penalizes confident wrong predictions. Sensitive to calibration.

## Top-k accuracy

An example is correct if fewer than `k` classes score strictly higher than its true class. Ties at the boundary count as correct.

## Confusion matrix

Rows = true class, columns = predicted class. Entry `(i, j)` is the count of true `i` predicted as `j`. The entries sum to the number of examples. From it you can derive precision, recall, and F1 per class.

## Bias vs variance

- High bias: too simple, underfits. Training and validation error are both high.
- High variance: chases the sample, overfits. Training error is low, validation error is high.

## Cross-validation

- **K-fold**: every sample is tested exactly once. Stratify when a class is rare.
- **Leave-one-out**: k = n. Almost unbiased but high variance and expensive.
- **Time series**: do not shuffle. Use a rolling or expanding window.
- Fit preprocessors inside the CV loop (or use a Pipeline) to avoid leakage.

## Hyperparameter search

- GridSearchCV: exhaustive, slow on large spaces.
- RandomizedSearchCV: samples a fixed number of combinations, usually good enough.
- Never tune on the final test set. Keep a completely untouched hold-out.

## Coding reference

Top-k accuracy and confusion matrix: [interview/coding_questions.md](../interview/coding_questions.md).
