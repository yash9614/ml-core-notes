# Logistic Regression

Source: codebasics multiclass dry-bean exercise + Krish Naik materials.  
**Exercise notebook:** [logistic_assignment.ipynb](04-logistic-regression/logistic_assignment.ipynb)

## What it is

A linear model that outputs a probability of the positive class. The linear score is passed through a sigmoid so the prediction stays in (0, 1). The decision boundary is still a hyperplane.

## Sigmoid

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Numerically stable form** (avoids overflow for large |z|):

\[
\sigma(z) = \frac{1}{2}\bigl(1 + \tanh(z/2)\bigr)
\]

## Loss

Binary cross-entropy (log loss). Do not use squared error on the probability.

**From logits** (stable, no intermediate sigmoid):

\[
\begin{aligned}
\ell(y=1, z) &= \log(1 + e^{-z}) \\
\ell(y=0, z) &= \log(1 + e^{z})
\end{aligned}
\]

## Coefficients

A coefficient is a change in log-odds. `exp(w_j)` is an odds ratio.

## Multiclass

- One-vs-rest: one binary model per class.
- Softmax (multinomial): one joint model. Probabilities sum to 1.

sklearn’s `LogisticRegression` handles multiclass automatically (default is `multi_class='auto'`).

### Practical multiclass example (dry beans)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_excel("dry_bean_dataset.xlsx")
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=300, C=0.5, class_weight="balanced", tol=0.001))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Hyperparameter tuning pattern (from Krish materials)

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

param_grid = {
    "C": [0.01, 0.1, 1.0, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best params (Grid):", grid.best_params_)

# Faster alternative
random_search = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=cv,
    random_state=42
)
random_search.fit(X_train, y_train)
print("Best params (Random):", random_search.best_params_)
```

## Threshold

0.5 is not sacred. Tune it on a precision-recall curve when classes are imbalanced or error costs differ.

## sklearn tips

- Default penalty is L2. `C` is the inverse regularization strength.
- Use a pipeline so scaling is fit only on train.
- `class_weight='balanced'` is useful for imbalanced multiclass problems.
- Prefer RandomizedSearchCV when the search space is large.

## Interview questions

1. Why is log loss used instead of MSE for logistic regression?
2. Explain the difference between one-vs-rest and softmax for multiclass logistic regression.
3. What does the coefficient in logistic regression represent? How do you interpret an odds ratio?
4. When would you set `class_weight='balanced'`?
5. Why is scaling important before logistic regression with L2 regularization?
6. How do you choose the decision threshold in an imbalanced classification problem?
7. Walk through a multiclass logistic regression pipeline with sklearn (including evaluation).
8. What is the difference between `C` and the regularization strength \(\lambda\)?
9. How would you tune a logistic regression model? When would you prefer RandomizedSearchCV over GridSearchCV?
