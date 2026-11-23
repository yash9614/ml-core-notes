# Ensemble Learning – Key Takeaways

Source: codebasics.io Takeaways – Ensemble Learning.

Ensemble methods combine multiple models to get better accuracy and robustness than any single model.

## Basic techniques

- Majority vote (classification)
- Average / weighted average (regression)
- Hard voting: majority class  
- Soft voting: average predicted probabilities (better when models are well-calibrated)

## Advanced techniques

### Bagging (Bootstrap Aggregating)

- Train many models on random subsets (with replacement) of the data.
- Aggregate predictions (vote or average).
- Reduces variance, robust to outliers, handles high dimensionality.
- **Random Forest**: bagging + random feature subsets at each split. Extremely popular for both classification and regression.

### Boosting

Sequential: each new model focuses on the errors of the previous ones.

- **AdaBoost**: increases weights of misclassified instances. Uses decision stumps (depth-1 trees).
- **Gradient Boosting**: fits new trees on the residual errors (pseudo-residuals). Learning rate controls how much each tree contributes.
  - Classification uses logistic loss; regression uses MSE.
- **XGBoost**: highly optimized gradient boosting.  
  Benefits: high accuracy, built-in regularization, handles missing values, feature importance, very fast (pruning, parallel processing, cache awareness, sparse data).

### Stacking

Train a meta-model on the predictions of base models (not covered in depth in the takeaways).

## Practical notes

- Random Forest and Gradient Boosting / XGBoost are usually among the strongest classical algorithms.
- All trees in a Random Forest use the same splitting criterion; you tune number of trees, max depth, min samples, etc.
- Boosting can overfit if not regularized; early stopping and learning rate help.
