# Model Evaluation & Fine-Tuning – Key Takeaways

Source: codebasics.io Takeaways – Model Evaluation & Fine Tuning.

## ROC Curve & AUC

- **ROC**: plot of True Positive Rate vs. False Positive Rate at different classification thresholds.
- **AUC**: area under the ROC curve.  
  - 0.5 = random  
  - 1.0 = perfect separation  
- When talking to stakeholders, translate ROC/AUC/precision into business impact (cost of false positives vs. false negatives).

## Cross-Validation

### K-Fold

1. Split data into K folds.  
2. Train on K-1 folds, test on the remaining fold.  
3. Repeat so every fold is used as test once.  
4. Average the scores.

Also useful for comparing different hyperparameter settings of the same model.

### Stratified K-Fold

Same as K-Fold but preserves the class distribution in every fold. Important for imbalanced classification.

## Hyperparameter Tuning

- **GridSearchCV**: exhaustive search over a defined grid of parameters, evaluated with cross-validation.
- **RandomizedSearchCV**: samples a fixed number of random combinations (`n_iter`). Much faster when the search space is large. With high `n_iter` it approaches GridSearchCV behavior.

Always prefer RandomizedSearchCV for large grids; use GridSearchCV only for small, carefully chosen spaces.

## Choosing the Right Metric

**Classification**  
- Accuracy: overall correctness (good when balanced).  
- Precision: quality of positive predictions (costly false positives).  
- Recall: coverage of actual positives (costly false negatives).  
- AUC-ROC: ranking quality.  
- F1: balance of precision and recall.  
- Confusion matrix: see exactly where errors occur.

**Regression**  
- MSE: penalizes large errors.  
- RMSE: same units as target, easier to interpret.  
- \( R^2 \): proportion of variance explained.

## Model Selection (high-level)

No single perfect flowchart. Consider:  
- Problem type (regression / classification / clustering)  
- Relationships between features (linear vs. non-linear)  
- Data type (tabular, text, image)  
- Data size and compute budget  

Judgment + experimentation + cross-validation is the practical path.
