# Supervised ML: Regression – Key Takeaways

Source: codebasics.io Takeaways – Supervised Machine Learning: Regression.

## Linear Regression

**Single variable**:  
\[ y = mx + b \]  
- \( m \): slope  
- \( b \): intercept  

Establishes a linear relationship between one independent variable and the dependent variable.

**Multiple variables**:  
\[ y = m_1 x_1 + m_2 x_2 + \dots + b \]

`model.predict()` applies the learned coefficients and intercept.

## Cost Function

- Error / Loss = predicted − actual.
- **MAE**: average of absolute errors (robust to outliers in direction).
- **MSE**: average of squared errors. Penalizes large errors more. Standard cost for linear regression because it is differentiable and convex.

Why prefer MSE for gradient descent?  
- Sensitive to outliers (sometimes desirable).  
- Continuously differentiable.  
Use MAE only when outliers are extreme and you want robustness.

## Gradient Descent

Optimization method that iteratively updates slope (\( m \)) and intercept (\( b \)) to minimize the cost (usually MSE).

- Learning rate and number of epochs control the process.
- In practice you rarely implement it from scratch; libraries do it. Understanding is still important for interviews and debugging.

Key supporting ideas:  
- Derivative: slope of the tangent.  
- Partial derivatives: change w.r.t. one variable while holding others fixed.  
- Chain rule: used when the cost depends on composed functions.

## Train / Test Split & Metrics

- Split data so the model is evaluated on unseen data (`train_test_split`, `random_state` for reproducibility).
- Common metrics: MSE, MAE, \( R^2 \).  
- \( R^2 \) is easier to interpret (proportion of variance explained).

## Data Preprocessing

- **One-hot encoding**: for nominal categoricals (no order).  
  - Creates binary columns.  
  - Drop one column to avoid multicollinearity (`drop_first=True` in `pd.get_dummies`).
- Label encoding: for ordinal categories that have order.

## Polynomial Regression

\[ y = b_0 + b_1 x + b_2 x^2 + \dots + b_n x^n \]

- Degree 1 = ordinary linear regression.  
- Degree chosen by domain knowledge + cross-validation / trial-and-error.

## Overfitting & Underfitting

- **Overfitting**: model memorizes noise; poor on new data.  
  Causes: too complex model, too many features, insufficient data, no regularization, poor validation.  
  Remedies: simpler model, more data, regularization, better feature selection, proper validation.
- **Underfitting**: model too simple; fails even on training data.
- **Balanced fit**: good on both train and unseen data.

## Regularization

- L2 → Ridge  
- L1 → Lasso  
- Helps control overfitting when features are many.  
- Alpha (regularization strength) tuned via cross-validation.

## Bias–Variance Tradeoff

- High bias → underfitting (too simple).  
- High variance → overfitting (too sensitive to training data).  
- Goal: low bias + low variance.
