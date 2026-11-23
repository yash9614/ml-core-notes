# Ensembles

Source: codebasics ensemble exercises (Random Forest, Gradient Boosting, XGBoost) + core concepts.  
**Industrial relevance added:** why gradient boosting dominates tabular data in production, practical tuning, and common pitfalls.

**Exercise notebook:** [ensemble_ex1_assignment.ipynb](09-ensembles/ensemble_ex1_assignment.ipynb)

## Bagging and Random Forests

Bagging fits the same model on bootstrap samples and averages them. Variance drops. Bias stays about the same.

A random forest is bagged trees plus a random subset of features at each split. The feature subset decorrelates the trees.

**Out-of-bag (OOB) evaluation**: each tree misses about a third of the rows. Score those rows with the trees that did not see them. A free estimate, not a substitute for a final holdout if you also tuned hyperparameters on OOB.

### Practical code (cancer data – Decision Tree vs Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# After loading and preparing X, y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Decision Tree")
print(classification_report(y_test, dt.predict(X_test)))

rf = RandomForestClassifier(n_estimators=50, max_features="log2", criterion="entropy", random_state=42)
rf.fit(X_train, y_train)
print("Random Forest")
print(classification_report(y_test, rf.predict(X_test)))
```

## Boosting

Boosting adds models sequentially. Each one targets what the current sum gets wrong.

### AdaBoost
Reweights misclassified samples. Sensitive to label noise.

### Gradient Boosting
Fits the next tree to the negative gradient of the loss (residuals for squared error). Learning rate multiplies each tree’s contribution.

### XGBoost / LightGBM / CatBoost
Highly optimized gradient boosting with regularization, missing-value handling, and fast training. Default strong baseline for most tabular problems in industry.

### Practical code (Gradient Boosting + XGBoost)

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print("GBM accuracy:", accuracy_score(y_test, gb.predict(X_test)))

xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, colsample_bytree=0.5, random_state=42)
xgb.fit(X_train, y_train)
print("XGBoost accuracy:", accuracy_score(y_test, xgb.predict(X_test)))
```

## Industrial relevance

- **Gradient boosting (especially XGBoost / LightGBM / CatBoost) is the default strong baseline** for structured/tabular data in production and competitions.
- Random Forest is preferred when you need robustness, less tuning, and good feature importance with lower risk of overfitting.
- Single deep trees are rarely deployed alone; ensembles or shallow trees are preferred.
- Always use early stopping and proper cross-validation with boosting to avoid overfitting.
- Feature importance from tree ensembles is widely used; SHAP is preferred for rigorous explanations.
- In production: prefer fewer well-regularized trees for latency; monitor for data drift; retrain periodically.
- No scaling required for tree-based models (unlike linear models or SVM).

## Interview questions

1. What is the difference between bagging and boosting? Which reduces variance and which reduces bias?
2. Why is a Random Forest less prone to overfitting than a single decision tree?
3. How does gradient boosting work at a high level?
4. When would you choose XGBoost over Random Forest in production?
5. What is early stopping and why is it important for boosting algorithms?
6. Explain the role of learning rate in gradient boosting.
7. How do you interpret feature importance from a tree-based ensemble? What are its limitations?
8. What is out-of-bag evaluation and when is it reliable?
9. Why do tree ensembles not require feature scaling?
10. Describe a situation where you would prefer a simple logistic regression over a gradient boosting model.
