# Ensembles

Source: codebasics ensemble exercises + Krish Naik materials (AdaBoost, Gradient Boosting, XGBoost).  
**Industrial relevance and intuition added.**

**Exercise notebook:** [ensemble_ex1_assignment.ipynb](09-ensembles/ensemble_ex1_assignment.ipynb)

## Bagging and Random Forests

Bagging fits the same model on bootstrap samples and averages them. Variance drops. Bias stays about the same.

A random forest is bagged trees plus a random subset of features at each split. The feature subset decorrelates the trees.

**Out-of-bag (OOB) evaluation**: each tree misses about a third of the rows. Score those rows with the trees that did not see them. A free estimate, not a substitute for a final holdout if you also tuned hyperparameters on OOB.

## Boosting – Intuition

Boosting builds models sequentially. Each new model focuses on the mistakes of the previous ones.

### AdaBoost

Each training sample starts with equal weight. After each weak learner (usually a shallow tree):
1. Compute the weighted error.
2. Give the learner a weight α = 0.5 × log((1 − err) / err). Better learners get higher weight.
3. Increase the weight of misclassified samples and decrease the weight of correctly classified ones.
4. Renormalize.

The final prediction is a weighted vote. AdaBoost is sensitive to label noise because noisy points keep receiving higher weight.

**Practical note**: Works well on moderately sized tabular data. Often outperformed by modern gradient boosting on large datasets, but still useful for understanding boosting.

### Gradient Boosting

Fits the next tree to the negative gradient of the loss (residuals for squared error). Learning rate multiplies each tree’s contribution so many small steps beat one large step.

### XGBoost / LightGBM / CatBoost

Highly optimized gradient boosting with regularization, second-order information, missing-value handling, and fast training. The default strong baseline for most tabular problems in industry.

### Practical code

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
ada.fit(X_train, y_train)
print("AdaBoost")
print(classification_report(y_test, ada.predict(X_test)))

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print("GBM")
print(classification_report(y_test, gb.predict(X_test)))

# XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
print("XGBoost")
print(classification_report(y_test, xgb.predict(X_test)))
```

## Industrial relevance

- **Gradient boosting (XGBoost / LightGBM / CatBoost) is the default strong baseline** for structured/tabular data in production and competitions.
- Random Forest is preferred when you need robustness, less tuning, and good feature importance with lower risk of overfitting.
- Single deep trees are rarely deployed alone; ensembles or shallow trees are preferred.
- Always use early stopping and proper cross-validation with boosting to avoid overfitting.
- Feature importance from tree ensembles is widely used; SHAP is preferred for rigorous explanations.
- In production: prefer fewer well-regularized trees for latency; monitor for data drift; retrain periodically.
- No scaling required for tree-based models.

## Interview questions

1. What is the difference between bagging and boosting? Which reduces variance and which reduces bias?
2. How does AdaBoost reweight samples? Why is it sensitive to label noise?
3. How does gradient boosting work at a high level?
4. When would you choose XGBoost over Random Forest in production?
5. What is early stopping and why is it important for boosting algorithms?
6. Explain the role of learning rate in gradient boosting.
7. How do you interpret feature importance from a tree-based ensemble? What are its limitations?
8. What is out-of-bag evaluation and when is it reliable?
9. Why do tree ensembles not require feature scaling?
10. Describe a situation where you would prefer a simple logistic regression over a gradient boosting model.
