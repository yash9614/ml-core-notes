# Linear regression

Source: `1-Simple+Linear+Regression.pdf`, `2-Multiple+Linear+Regression.pdf`, `linearregressionOLS.pdf`, `height-weight.csv`, `economic_index.csv`, `Multiple+Linear+Regression-+Economics+Dataset.ipynb`.

## What it is

Fit a straight line, or a flat hyperplane, that predicts a number.

\[
\hat{y} = w_0 + w_1 x_1 + \cdots + w_p x_p
\]

Simple linear regression has one input. Multiple linear regression has many. The word linear means linear in the weights, not that the raw feature must be a straight line of the original column.

## How the weights are found

Ordinary least squares minimizes the sum of squared residuals.

\[
J(w) = \sum_i (y_i - \hat{y}_i)^2
\]

Closed form, when \(X^\top X\) is invertible:

\[
w = (X^\top X)^{-1} X^\top y
\]

Gradient descent does the same job when the matrix is large or singular. Learning rate too big diverges. Too small just wastes steps.

## Assumptions that actually matter

- The relationship is roughly linear in the features you gave the model.
- Errors have constant spread (homoscedasticity). A fan shape means the intervals are wrong.
- Errors are independent. Time series often breaks this.
- For the usual confidence intervals, errors are roughly normal. The point prediction is more robust than the interval.
- Features should not be exact copies of each other. Multicollinearity makes individual weights unstable even when predictions are fine.

## Reading the fit

- Coefficient: change in \(y\) for a one-unit change in that feature, holding the others fixed. Only meaningful after you know the units and whether features were scaled.
- \(R^2\): fraction of variance explained. It rises when you add junk features. Adjusted \(R^2\) penalizes that.
- Residual plot: curved pattern means the linear form is wrong. Funnel shape means variance is not constant.

## Interview questions

1. Why squared error and not absolute error? Squared error is differentiable and punishes large misses. Absolute error is the median regression and is harder to optimize, but more robust to outliers.
2. What does a large positive coefficient not mean? It does not mean the feature caused the outcome. It is an association conditional on the other columns.
3. Multicollinearity: predictions can still be good. Individual coefficients cannot be trusted. Check VIF or drop one of the correlated columns.
4. How do you know the line is the wrong shape? Residual plot bends. Then try a transform or polynomial features, not a deeper story about the same line.

## Kaggle

- [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques): start with a linear model before any tree. Metric is RMSLE.
- [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance): small, good for coefficients and residual plots.
- Course data: `height-weight.csv` for the one-feature case, `economic_index.csv` for the multiple case.
