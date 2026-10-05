# Metrics, overfitting, cross-validation

Source: performance-metrics PDFs, `MSE,RMSE,MAE.pdf`, `8-Overfitting+And+Underfitting.pdf`, `Types+Of+Cross+Validation.pdf`.

## Regression metrics

- MAE: average absolute error. Same unit as y. Robust to a few huge misses.
- MSE: average squared error. Punishes large misses. Unit is y squared.
- RMSE: square root of MSE. Same unit as y, still punishes large misses.
- RMSLE: RMSE on the log. Used when a proportional miss matters. House Prices uses this.
- \(R^2\): variance explained. Can be negative if you lose to predicting the mean.

## Classification metrics

- Accuracy: fraction correct. Only if classes are balanced and errors cost the same.
- Precision: of the predicted positives, how many were positive.
- Recall: of the actual positives, how many you caught.
- F1: harmonic mean of precision and recall. Useful, not a business metric.
- ROC-AUC: ranking quality across thresholds. Can look strong when positives are rare.
- PR-AUC: precision against recall. The one to quote for fraud, churn, and disease.
- Log loss: punishes confident wrong probabilities.

Confusion matrix first. A single number second.

## Overfit and underfit

Underfit: high train error, high test error. Model is too simple, or features are missing.
Overfit: low train error, high test error. Model memorized. Constrain it, get more data, or clean leakage.

A learning curve separates the two. If both curves are high and close, add features or a richer model. If train is low and validation is high, regularize or get more rows.

## Cross-validation

- Holdout: one split. Fast, noisy.
- K-fold: rotate the validation fold. Stratify for rare classes.
- Time series: never shuffle. Train on the past, validate on the future.
- Group: keep a user or a hospital entirely in train or entirely in test.

Fit scalers, encoders, and feature selection inside the fold. Fitting them on all rows before the split is leakage.

## Interview questions

1. RMSE vs MAE? RMSE punishes big errors more.
2. Precision vs recall when a false alarm is expensive? Precision. When a miss is expensive, recall.
3. Why can ROC-AUC lie on fraud data? A model can rank most negatives below most positives and still flood you with false alarms, because negatives are almost the whole set.
4. Where does leakage hide? Target encoded with the full column. Scaler fit before the split. A feature that is only known after the event.

## Kaggle

Every problem in the Kaggle list. Name the metric before fitting, then check one leaked feature you almost used.
