# Supervised ML: Regression – Key Takeaways

Source: codebasics.io Takeaways – Supervised Machine Learning: Regression.  
**Industrial relevance added:** real-world usage, practical choices, and interview depth.

## Linear Regression

**Single variable**:  
\[ y = mx + b \]  
- \( m \): slope  
- \( b \): intercept  

**Multiple variables**:  
\[ y = m_1 x_1 + m_2 x_2 + \dots + b \]

`model.predict()` applies the learned coefficients and intercept.

**Industrial relevance:**  
- Still widely used as a strong, interpretable baseline (especially in finance, pricing, forecasting).  
- Prefer it when you need coefficients stakeholders can understand and defend.  
- Always check residual plots and linearity assumptions in practice.

## Cost Function

- Error / Loss = predicted − actual.
- **MAE**: average of absolute errors (robust to outliers).
- **MSE**: average of squared errors. Penalizes large errors more. Standard cost for linear regression because it is differentiable and convex.

Why prefer MSE for gradient descent?  
- Continuously differentiable.  
- Sensitive to outliers (sometimes desirable).  
Use MAE / Huber when outliers are extreme and you want robustness.

## Gradient Descent

Optimization method that iteratively updates slope (\( m \)) and intercept (\( b \)) to minimize the cost (usually MSE).

- Learning rate and number of epochs control the process.
- In practice you rarely implement it from scratch; libraries do it. Understanding is still important for interviews and debugging.

Key supporting ideas:  
- Derivative: slope of the tangent.  
- Partial derivatives: change w.r.t. one variable while holding others fixed.  
- Chain rule: used when the cost depends on composed functions.

## Train / Test Split & Metrics

- Split data so the model is evaluated on unseen data (`train_test_split`, `random_state` for reproducibility). Prefer time-based or group-based splits when data has temporal or entity structure.
- Common metrics: MSE, MAE, RMSE, \( R^2 \), MAPE.  
- \( R^2 \) is easier to interpret (proportion of variance explained). RMSE is preferred in business contexts because it is in the same units as the target.

**Industrial relevance:**  
- Always report business-relevant error (e.g., “average absolute error of ₹500 on price prediction”).  
- Watch for data leakage in the split.

## Data Preprocessing

- **One-hot encoding**: for nominal categoricals (no order). Drop one column to avoid multicollinearity.
- Label encoding: for ordinal categories that have order.
- Scaling is important for gradient-based methods and regularized models.

## Polynomial Regression

\[ y = b_0 + b_1 x + b_2 x^2 + \dots + b_n x^n \]

- Degree 1 = ordinary linear regression.  
- Degree chosen by domain knowledge + cross-validation. High degrees overfit quickly.

## Overfitting & Underfitting

- **Overfitting**: model memorizes noise; poor on new data.  
- **Underfitting**: model too simple; fails even on training data.
- **Balanced fit**: good on both train and unseen data.

Remedies for overfitting: more data, regularization, simpler model, better feature selection, proper validation, early stopping.

## Regularization

- L2 → Ridge (shrinks coefficients, rarely zeros them)
- L1 → Lasso (can zero out coefficients → feature selection)
- Helps control overfitting when features are many.  
- Alpha (regularization strength) tuned via cross-validation.

**Industrial relevance:**  
- Ridge is often preferred when you want to keep all features.  
- Lasso is useful for high-dimensional problems where interpretability matters.

## Bias–Variance Tradeoff

- High bias → underfitting (too simple).  
- High variance → overfitting (too sensitive to training data).  
- Goal: low bias + low variance.

## Interview Questions

1. Why is MSE preferred over MAE as a loss function for linear regression?
2. Explain the bias-variance tradeoff with a concrete example.
3. When would you choose Ridge over Lasso (and vice versa)?
4. How do you diagnose and fix overfitting in a regression model?
5. What is the difference between \( R^2 \) and adjusted \( R^2 \)? When does it matter?
6. Walk through how gradient descent updates the parameters of a linear regression model.
7. How would you handle a regression problem where the target has heavy outliers?
