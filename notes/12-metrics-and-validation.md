# Metrics and Validation

Source: codebasics model evaluation exercises (heart failure + hotel reservations) + core concepts.  
**Industrial relevance, intuition, and practical patterns added.**

## Classification Metrics – Intuition

- **Accuracy**: fraction of correct predictions. Misleading when one class dominates (e.g., 99% accuracy on 1% fraud is useless).
- **Precision** = TP / (TP + FP): of the things you flagged as positive, how many were actually positive? Critical when false alarms are expensive (spam filtering, medical alerts).
- **Recall** = TP / (TP + FN): of all actual positives, how many did you catch? Critical when misses are expensive (fraud, disease detection, safety).
- **F1**: harmonic mean of precision and recall. Good single number when you care about both.
- **ROC-AUC**: measures ranking quality across all thresholds. Can look good even on rare positives.
- **PR-AUC**: focuses on the positive class. Prefer this for fraud, churn, and rare-event problems.
- **Log loss**: penalizes confident wrong predictions. Sensitive to calibration.

### Practical pattern (heart failure – compare models with ROC)

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

models = {
    "GaussianNB": GaussianNB(),
    "SVM-Linear": SVC(kernel="linear", probability=True),
    "XGBoost": XGBClassifier(random_state=42)
}

plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name} AUC: {auc:.3f}")
    print(classification_report(y_test, model.predict(X_test)))
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curves")
plt.show()
```

## Cross-Validation – Intuition

One train/test split can be lucky or unlucky. K-fold averages performance across multiple splits.

- **K-Fold**: every sample is tested exactly once. Good default.
- **Stratified K-Fold**: preserves class distribution in every fold. Essential for imbalanced data.
- **Time series**: never shuffle. Use rolling or expanding windows.

### Practical pattern (hotel – K-Fold vs Stratified)

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

kf = KFold(n_splits=7, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

nb = GaussianNB()
print("K-Fold accuracy:", cross_val_score(nb, X, y, cv=kf, scoring="accuracy").mean())
print("Stratified K-Fold accuracy:", cross_val_score(nb, X, y, cv=skf, scoring="accuracy").mean())
```

## Hyperparameter Search – Intuition

- **GridSearchCV**: tries every combination. Exhaustive but slow on large spaces.
- **RandomizedSearchCV**: samples a fixed number of combinations. Much faster and usually good enough.
- Always keep a completely untouched final test set. Never tune on it.

### Practical pattern

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

# Faster approach
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    random_state=42
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best score:", search.best_score_)
```

## Industrial relevance

- Define the primary metric with the business **before** modeling. Translate model metrics into cost or revenue impact.
- For imbalanced problems (fraud, medical, churn), report precision, recall, F1, or PR-AUC — never rely on accuracy alone.
- Always use stratified splits and pipelines so preprocessing is fit only on training data.
- Prefer RandomizedSearchCV (or Optuna) over full GridSearchCV for large search spaces.
- Monitor both model metrics and business metrics in production. Segment performance by customer type, geography, or time.
- Threshold selection is a business decision, not a modeling one. Tune it on a validation set using cost trade-offs.

## Interview questions

1. Why is accuracy a bad metric for imbalanced classification? What should you use instead?
2. Explain precision vs recall. Give a business example for each.
3. What is the difference between ROC-AUC and PR-AUC? When does PR-AUC matter more?
4. Why is Stratified K-Fold preferred over regular K-Fold for classification?
5. What is the difference between GridSearchCV and RandomizedSearchCV? Which would you use in practice and why?
6. How do you choose the decision threshold in a real system?
7. What is data leakage in cross-validation and how do you prevent it?
8. Walk through comparing three models using ROC curves and classification reports.
9. How would you evaluate a model that will be used for real-time fraud detection?
