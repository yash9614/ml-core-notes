# Linear Regression

Source: Krish Naik materials (economic index, practical notebooks) + core concepts.  
**Industrial relevance and practical patterns added.**

**Exercise notebook:** [regression_ex1.ipynb](02-linear-regression/regression_ex1.ipynb)

## What it is

Fit a straight line (or hyperplane) that predicts a continuous number.

\[
\hat{y} = w_0 + w_1 x_1 + \cdots + w_p x_p
\]

Simple linear regression has one input. Multiple linear regression has many. “Linear” means linear in the **weights**, not that the raw feature must be a straight line.

## How the weights are found

Ordinary least squares minimizes the sum of squared residuals:

\[
J(w) = \sum_i (y_i - \hat{y}_i)^2
\]

Closed form (when \(X^\top X\) is invertible):

\[
w = (X^\top X)^{-1} X^\top y
\]

Gradient descent is used when the matrix is large or singular.

### Practical code

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Example: economic index or similar tabular data
df = pd.read_csv("economic_index.csv")
df = df.drop(columns=["Unnamed: 0", "year", "month"], errors="ignore")

X = df.drop("target_column", axis=1)   # adjust column name
y = df["target_column"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
```

## Assumptions that actually matter

- Relationship is roughly linear in the features you gave the model.
- Errors have roughly constant spread (homoscedasticity).
- Errors are independent (time series often breaks this).
- Features should not be perfect copies of each other (multicollinearity makes individual weights unstable).

## Reading the fit

- Coefficient: change in \(y\) for a one-unit change in that feature, holding others fixed.
- \(R^2\): fraction of variance explained. Adjusted \(R^2\) penalizes extra features.
- Residual plot: curved pattern → linear form is wrong. Funnel shape → variance not constant.

## Regression metrics – quick reminder

- **MAE**: average absolute error. Robust to outliers. Same units as target.
- **MSE / RMSE**: penalizes large errors more. RMSE is in the same units as the target.
- **R²**: fraction of variance explained. Can be misleading if you keep adding features.

## Industrial relevance

- Still widely used as an interpretable baseline (pricing, risk scoring, forecasting when relationships are roughly linear).
- Coefficients can be explained to business stakeholders as “for every unit increase in X, Y changes by … holding other factors fixed.”
- Always check residual plots and VIF when the model will be used for interpretation rather than pure prediction.
- In production, monitor for multicollinearity changes and data drift that can make coefficients unstable.
- Prefer pipelines so scaling is consistent between training and serving.

## Interview questions

1. Why squared error and not absolute error?
2. What does a large positive coefficient **not** mean?
3. How do you diagnose and handle multicollinearity?
4. How do you know the linear form is wrong? What do you look for in residual plots?
5. Explain the difference between \(R^2\) and adjusted \(R^2\).
6. Walk through building a linear regression pipeline with scaling and evaluation.
7. When would you prefer MAE over RMSE?
8. How would you handle a linear regression model used for pricing when features are correlated?
