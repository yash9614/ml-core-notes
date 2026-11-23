# Linear regression

Source: codebasics exercises + takeaways.  
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

### Practical code (single variable – weather example)

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("weather_data.csv")
model = LinearRegression()
model.fit(df[['hours_sunlight']], df['daily_temperature'])

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Predict
hours = [[5], [8], [12]]
print(model.predict(hours))
```

### Multiple variables + train/test

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df[['hours_sunlight', 'humidity_level']]
y = df['daily_temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
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

## Interview questions

1. Why squared error and not absolute error? Squared error is differentiable and punishes large misses. Absolute error is more robust to outliers but harder to optimize.
2. What does a large positive coefficient **not** mean? It does not mean the feature caused the outcome. It is an association conditional on the other columns.
3. Multicollinearity: predictions can still be good. Individual coefficients cannot be trusted. Check VIF or drop one of the correlated columns.
4. How do you know the line is the wrong shape? Residual plot bends. Then try a transform or polynomial features.
5. Walk through a simple linear regression implementation with sklearn (or from scratch with gradient descent).
6. What is the difference between \(R^2\) and adjusted \(R^2\)? When does it matter?
7. How would you diagnose and handle multicollinearity in a linear model used for interpretation (e.g., pricing or risk)?
8. Explain training-serving skew in the context of linear regression features.
