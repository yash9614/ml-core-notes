# Polynomial features, Ridge, Lasso, Elastic Net

Source: codebasics L1/L2 tutorial + exercises.  
**Exercise notebook:** [l1_l2_regularization.ipynb](03-regularization/l1_l2_regularization.ipynb)

## Polynomial regression

Still a linear model. You add columns such as \(x^2\) and \(x_1 x_2\), then fit ordinary weights. The curve bends. The math does not.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25, random_state=5)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

Degree too high memorizes the sample. Use a validation curve on the degree, not the training \(R^2\).

## Why regularize

OLS will use a huge positive weight and a huge negative weight on two correlated features. Predictions can still look fine. The weights are nonsense, and a small data shift breaks them.

Add a penalty so weights stay small.

- **Ridge (L2)**: penalty \(\lambda \|w\|_2^2\). Shrinks weights toward zero. Does not set them to zero. Better when many features matter a little.
- **Lasso (L1)**: penalty \(\lambda \|w\|_1\). Can set weights to exactly zero. Does feature selection. Unstable when features are correlated: it picks one and drops the rest.
- **Elastic Net**: both penalties. The usual choice on wide, correlated tables.

\(\lambda\) is the strength. sklearn exposes `alpha` on these models.

**Always scale features** before Ridge or Lasso. The penalty treats a weight of 3 on age the same as a weight of 3 on income.

### Practical code

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge R²:", ridge.score(X_test, y_test))
print("Ridge coefs:", ridge.coef_)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print("Lasso R²:", lasso.score(X_test, y_test))
print("Lasso coefs (zeros show feature selection):", lasso.coef_)
```

## Interview questions

1. Ridge vs Lasso in one line? Ridge shrinks. Lasso can zero out.
2. Why scale? The penalty is on the weight, and the weight’s size depends on the feature’s unit.
3. Is polynomial regression nonlinear? Nonlinear in \(x\), linear in the weights.
4. When would you choose Elastic Net over pure Lasso or Ridge?
5. How do you choose the regularization strength (`alpha`)? (Cross-validation / validation curve.)
6. What happens to coefficients as `alpha` → ∞ in Ridge? In Lasso?
7. Explain a situation where Lasso is preferable to Ridge for a business problem.
8. How does regularization help with multicollinearity?
