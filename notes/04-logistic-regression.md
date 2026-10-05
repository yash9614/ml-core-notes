# Logistic regression

Source: `5-Logistic+Regression.pdf`, `Logistic+Regression+Implementation.ipynb`, `5.3-Performance+Metrics.pdf`.

## What it is

A linear model for a class probability. The line is passed through a sigmoid so the output stays in \((0, 1)\).

\[
P(y=1 \mid x) = \sigma(w \cdot x) = \frac{1}{1 + e^{-w \cdot x}}
\]

The decision boundary is the set of points where \(w \cdot x = 0\). That boundary is linear. The probability is not a class label until you pick a threshold, usually 0.5, and usually the wrong default if the classes are unbalanced.

## Loss

Do not use squared error on the probability. Use log loss (binary cross-entropy).

\[
J = -\frac{1}{n}\sum_i \left[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \right]
\]

There is no closed form. Training is gradient descent, Newton, or a solver such as lbfgs.

## Odds and coefficients

A coefficient is a change in log-odds. \(e^{w_j}\) is the odds ratio for a one-unit increase in feature \(j\). That is the sentence interviewers want, not "it is a probability weight."

## Regularization

sklearn's `LogisticRegression` regularizes by default (`C` is the inverse strength, `l2` by default). `C` small means a simpler boundary. This surprises people coming from a textbook that shows unregularized MLE.

## Multinomial

One-vs-rest trains one binary model per class. Multinomial / softmax trains one joint model. Softmax is the natural extension of the sigmoid.

## Interview questions

1. Why not linear regression for a yes/no label? Predictions fall outside 0 and 1, and outliers yank the line. Logistic maps to a probability and uses a proper classification loss.
2. Is the boundary linear? Yes, in the features you passed in. Polynomial features or a kernel make it nonlinear. The model itself did not grow a hidden layer.
3. Threshold 0.5 is wrong when? False positives and false negatives have different costs, or the positive class is rare. Tune the threshold on precision-recall, not accuracy.
4. How is this different from a sigmoid neuron? Same math for one layer. A network stacks many of them and learns the features. Logistic regression expects you to build the features.

## Kaggle

- [Titanic](https://www.kaggle.com/competitions/titanic): the standard first classification problem. Logistic regression should beat a blind "everyone dies" baseline before you touch a tree.
- [Porto Seguro safe driver prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction): imbalanced insurance risk, useful after Titanic. Metric is normalized Gini.
- Course path: the phishing project in `NETWORKsecurity` is a real classification problem from the same dump.
