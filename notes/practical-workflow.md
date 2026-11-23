# Practical End-to-End ML Workflow

Patterns extracted from Krish Naik notebooks (holiday package prediction, used car price, economic index, diabetes, and other practical exercises).

## Typical flow seen across the materials

1. **Load and inspect** data (`df.head()`, `df.info()`, `df.describe()`)
2. **Clean categorical values** (fix inconsistent labels like ‘Fe Male’ → ‘Female’)
3. **Handle missing values**
   - Numerical: median or mean
   - Categorical: mode
   - Check missing percentage first
4. **Drop low-value columns** (IDs, high-cardinality names that are not useful, pure noise)
5. **Encode categoricals** (one-hot or label encoding depending on cardinality)
6. **Scale numerical features** (StandardScaler or MinMaxScaler) — critical for distance-based and gradient-based models
7. **Train/test split** (stratify for classification)
8. **Train baseline models** (Linear/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting / XGBoost)
9. **Evaluate** with appropriate metrics (classification report / ROC-AUC for classification; MAE, RMSE, R² for regression)
10. **Tune** with GridSearchCV or RandomizedSearchCV
11. **Interpret** feature importance or SHAP when needed

## Practical missing-value pattern

```python
# Check missing
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])

# Impute
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

## Regression-specific notes

- Drop columns that are pure identifiers or have extremely high cardinality unless they are engineered into useful features.
- Always report MAE and RMSE (and R²) — MAE is often more interpretable to stakeholders.
- Check residual plots. Systematic patterns mean the linear form or the feature set is incomplete.
- For price prediction or forecasting tasks, tree-based models (Gradient Boosting / XGBoost) frequently outperform plain linear regression once non-linearities appear.

## Industrial relevance

- Always inspect data quality before modeling. Inconsistent categories and missing values are common in real datasets.
- Use pipelines so preprocessing is reproducible and leakage-free.
- Start with simple baselines before jumping to complex models.
- Document assumptions (how missing values were handled, why certain features were dropped).
- In production, the same preprocessing steps must be applied at inference time — pipelines or feature stores help.

## Interview questions

1. Walk through the steps you would take when you receive a new tabular dataset.
2. How do you decide between median and mean for imputing numerical features?
3. Why is it important to handle inconsistent categorical labels before encoding?
4. What is data leakage and how do you prevent it during preprocessing?
5. When would you use a Pipeline in sklearn?
6. For a used-car price prediction problem, what features would you drop or engineer, and which models would you try first?
