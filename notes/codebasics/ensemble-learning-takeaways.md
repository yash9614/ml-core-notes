# Ensemble Learning – Key Takeaways

Source: codebasics.io Takeaways – Ensemble Learning.  
**Industrial relevance added:** why ensembles dominate tabular data and practical production notes.

Ensemble methods combine multiple models to get better accuracy and robustness than any single model. In industry, gradient boosting ensembles are the default choice for most tabular problems.

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
- **Random Forest**: bagging + random feature subsets at each split. Extremely popular, stable, and hard to overfit. Excellent first strong baseline.

### Boosting

Sequential: each new model focuses on the errors of the previous ones.

- **AdaBoost**: increases weights of misclassified instances. Uses decision stumps (depth-1 trees).
- **Gradient Boosting**: fits new trees on the residual errors (pseudo-residuals). Learning rate controls how much each tree contributes.
  - Classification uses logistic loss; regression uses MSE.
- **XGBoost / LightGBM / CatBoost**: highly optimized gradient boosting.  
  Benefits: high accuracy, built-in regularization, handles missing values, feature importance, very fast (pruning, parallel processing, cache awareness, sparse data).

**Industrial relevance:**  
- XGBoost / LightGBM / CatBoost win the majority of Kaggle tabular competitions and are the workhorse in production for structured data.  
- Random Forest is preferred when you need less tuning and more robustness.  
- Always use early stopping and proper cross-validation with boosting to avoid overfitting.  
- Feature importance from tree ensembles is widely used for interpretation (though SHAP is better for rigorous explanations).

### Stacking

Train a meta-model on the predictions of base models. Powerful but more complex to maintain in production; used less often than pure boosting unless the lift is significant.

## Practical notes

- Random Forest and Gradient Boosting / XGBoost are usually among the strongest classical algorithms.
- All trees in a Random Forest use the same splitting criterion; you tune number of trees, max depth, min samples, etc.
- Boosting can overfit if not regularized; early stopping and learning rate help.
- In production, prefer fewer, well-regularized trees over extremely deep ensembles for latency and maintainability.

## Interview Questions

1. What is the difference between bagging and boosting? Which one reduces variance and which one reduces bias?
2. Why is Random Forest less prone to overfitting than a single decision tree?
3. How does gradient boosting work at a high level?
4. When would you choose XGBoost over Random Forest in production?
5. What is early stopping and why is it important for boosting algorithms?
6. Explain the role of learning rate in gradient boosting.
7. How do you interpret feature importance from a tree-based ensemble? What are its limitations?
