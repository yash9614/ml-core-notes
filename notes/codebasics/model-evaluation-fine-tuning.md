# Model Evaluation & Fine-Tuning – Key Takeaways

Source: codebasics.io Takeaways – Model Evaluation & Fine Tuning.  
**Industrial relevance added:** production evaluation practices, metric selection by business cost, and modern tuning approaches.

## ROC Curve & AUC

- **ROC**: plot of True Positive Rate vs. False Positive Rate at different classification thresholds.
- **AUC**: area under the ROC curve.  
  - 0.5 = random  
  - 1.0 = perfect separation  
- When talking to stakeholders, translate ROC/AUC/precision into business impact (cost of false positives vs. false negatives).

**Industrial relevance:**  
- AUC is good for ranking models but does not tell you the best operating threshold.  
- Always plot precision-recall curves when classes are imbalanced.  
- Business stakeholders rarely care about AUC; they care about “how many frauds did we catch at an acceptable false-positive rate?”

## Cross-Validation

### K-Fold

1. Split data into K folds.  
2. Train on K-1 folds, test on the remaining fold.  
3. Repeat so every fold is used as test once.  
4. Average the scores.

Also useful for comparing different hyperparameter settings of the same model.

### Stratified K-Fold

Same as K-Fold but preserves the class distribution in every fold. Important for imbalanced classification.

**Industrial relevance:**  
- Prefer time-series or group-based splits (GroupKFold) when data has temporal dependence or repeated entities (users, accounts). Random K-Fold can leak information.  
- Nested cross-validation is the gold standard when you also tune hyperparameters.

## Hyperparameter Tuning

- **GridSearchCV**: exhaustive search over a defined grid of parameters, evaluated with cross-validation.
- **RandomizedSearchCV**: samples a fixed number of random combinations (`n_iter`). Much faster when the search space is large.
- Modern alternatives: Optuna, Hyperopt, Ray Tune (Bayesian optimization, pruning).

Always prefer RandomizedSearchCV or Bayesian methods for large grids; use GridSearchCV only for small, carefully chosen spaces.

**Industrial relevance:**  
- Tune on a proper validation strategy, not just the training set.  
- Keep a simple baseline (default parameters) for comparison.  
- Document the final hyperparameters and the search space.

## Choosing the Right Metric

**Classification**  
- Accuracy: overall correctness (good when balanced).  
- Precision: quality of positive predictions (costly false positives).  
- Recall: coverage of actual positives (costly false negatives).  
- AUC-ROC: ranking quality.  
- F1: balance of precision and recall.  
- Confusion matrix: see exactly where errors occur.  
- Cost-sensitive / custom business metrics when the cost of errors is known.

**Regression**  
- MSE: penalizes large errors.  
- RMSE: same units as target, easier to interpret.  
- MAE: robust to outliers.  
- \( R^2 \): proportion of variance explained.  
- MAPE / business-specific error when relative error matters.

**Industrial relevance:**  
- Define the primary metric with the business before modeling.  
- Always report both model metrics and business impact metrics.  
- Segmented evaluation (by geography, customer segment, time) often reveals problems that overall metrics hide.

## Model Selection (high-level)

No single perfect flowchart. Consider:  
- Problem type (regression / classification / clustering)  
- Relationships between features (linear vs. non-linear)  
- Data type (tabular, text, image)  
- Data size and compute budget  
- Need for interpretability and latency constraints

Judgment + experimentation + proper validation is the practical path. In tabular data, start with a linear model and a gradient boosting model.

## Interview Questions

1. Why is AUC-ROC not always the best metric for an imbalanced classification problem?
2. What is the difference between K-Fold and Stratified K-Fold? When would you use each?
3. Explain the difference between GridSearchCV and RandomizedSearchCV. Which one would you use in practice and why?
4. How do you choose the right evaluation metric for a business problem?
5. What is nested cross-validation and when is it necessary?
6. How would you evaluate a model that will be used for real-time fraud detection?
7. What are common pitfalls in hyperparameter tuning?
