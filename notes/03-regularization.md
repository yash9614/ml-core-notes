# Polynomial features, Ridge, Lasso, Elastic Net

Source: `9-Polynomialregression.pdf`, `Ridge,Lasso+And+Elasticnet.pdf`, `Ridge Lassso Elastic Regression Practicals` on the Algerian forest-fires data.

## Polynomial regression

Still a linear model. You add columns such as \(x^2\) and \(x_1 x_2\), then fit ordinary weights. The curve bends. The math does not.

Degree too high memorizes the sample. Use a validation curve on the degree, not the training \(R^2\).

## Why regularize

OLS will use a huge positive weight and a huge negative weight on two correlated features. Predictions can still look fine. The weights are nonsense, and a small data shift breaks them.

Add a penalty so weights stay small.

- Ridge (L2): penalty \(\lambda \lVert w \rVert_2^2\). Shrinks weights toward zero. Does not set them to zero. Better when many features matter a little.
- Lasso (L1): penalty \(\lambda \lVert w \rVert_1\). Can set weights to exactly zero. Does feature selection. Unstable when features are correlated: it picks one and drops the rest.
- Elastic Net: both penalties. The usual choice on wide, correlated tables.

\(\lambda\) is the strength. sklearn exposes `alpha` on these models, and `C = 1/\lambda` on logistic regression. Same idea, inverted name.

Scale features before Ridge or Lasso. The penalty treats a weight of 3 on age the same as a weight of 3 on income.

## Interview questions

1. Ridge vs Lasso in one line? Ridge shrinks. Lasso can zero out.
2. Why scale? The penalty is on the weight, and the weight's size depends on the feature's unit.
3. Is polynomial regression nonlinear? Nonlinear in \(x\), linear in the weights.

## Kaggle

House Prices with Ridge or Elastic Net is the right second model after OLS. The Algerian fires notebook in the course dump is the local version.
