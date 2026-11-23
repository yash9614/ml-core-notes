# Decision Trees

Source: Krish Naik materials (diabetes regressor, practical implementations) + core concepts.  
**Industrial relevance and practical patterns added.**

## Split criteria

**Classification**
- Gini impurity: \(1 - \sum p_k^2\)
- Entropy: \(-\sum p_k \log p_k\)

**Regression**
Variance reduction (or absolute error). The leaf prediction is the mean (or median) of the targets in that leaf.

## Numeric splits

Sort the column. Candidate thresholds lie halfway between consecutive distinct values. Rows with feature ≤ threshold go left.

## Why trees overfit

A single tree can keep splitting until every leaf is pure. The top split also decides every split below it, so the tree is unstable (high variance).

## Stopping overfitting

- `max_depth`
- `min_samples_leaf` / `min_samples_split`
- Cost-complexity pruning (`ccp_alpha`)
- Ensemble methods (Random Forest, Gradient Boosting)

## Extrapolation

A tree cannot predict past the range of leaf means it saw. It produces a step function.

### Practical code – Decision Tree Regressor with hyperparameter tuning (diabetes style)

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assume X, y are prepared (e.g., diabetes dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [3, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", None]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
best_tree = grid.best_estimator_
y_pred = best_tree.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))
```

### Visualization and feature importance

```python
from sklearn import tree
import matplotlib.pyplot as plt

# Visualize (keep max_depth small for readability)
plt.figure(figsize=(15, 8))
tree.plot_tree(best_tree, filled=True, max_depth=3, feature_names=X.columns)
plt.show()

# Feature importance
importances = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))
```

## Industrial relevance

- Single decision trees are highly interpretable and still used in regulated domains (credit, insurance, healthcare) where stakeholders need to see the exact rules.
- Almost never the best pure predictive model — Random Forest or gradient boosting almost always outperform them.
- Useful for feature importance exploration and for generating business rules that can be implemented without a model serving layer.
- In production, prefer shallow trees (`max_depth` 3–6) or tree ensembles; deep single trees overfit and are brittle to data shifts.
- For regression, always compare against a simple linear baseline and check residual plots.
- Tree visualization is excellent for explaining decisions to non-technical stakeholders when the tree is kept shallow.

## Interview questions

1. How does a decision tree choose the best split?
2. Why do single decision trees overfit so easily? How do you control it?
3. Compare Gini impurity and entropy. Does the choice matter much in practice?
4. Why can’t a decision tree extrapolate beyond the training data range?
5. When would you deploy a single decision tree instead of a Random Forest?
6. Explain how `min_samples_leaf` and `max_depth` affect bias and variance.
7. Walk through building a Decision Tree pipeline with categorical encoding and evaluation.
8. What is cost-complexity pruning and when would you use it?
9. How would you tune a decision tree regressor? What metrics would you use?
10. How can you use a decision tree to explain model decisions to business stakeholders?
