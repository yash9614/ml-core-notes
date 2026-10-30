# Core ML Formulas

Highlighted key formulas from the notes, with short explanations. Focus on what the formula *does* and when it matters.

## Linear Regression

$$
\hat{y} = w_0 + w_1 x_1 + \cdots + w_p x_p = w^\top x + b
$$

Predicts a continuous value. Linear in the weights (you can still add polynomial features of $x$).

**Ordinary Least Squares (OLS) loss**

$$
J(w) = \sum_i (y_i - \hat{y}_i)^2 = \| y - Xw \|^2
$$

Minimizes squared residuals. Differentiable, penalizes large errors more, matches Gaussian noise assumption.

**Closed-form solution** (when $X^\top X$ is invertible)

$$
w = (X^\top X)^{-1} X^\top y
$$

Direct solution. Gradient descent is used when the matrix is large or singular.

**Gradient descent update**

$$
w := w - \alpha \nabla_w J(w)
$$

$\alpha$ is the learning rate. Too large → diverges; too small → slow.

---

## Logistic Regression

**Sigmoid (maps linear score to probability)**

$$
P(y=1 \mid x) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

Output stays in $(0,1)$. Decision boundary is still linear: where $w \cdot x + b = 0$.

**Log loss (binary cross-entropy)**

$$
J = -\frac{1}{n} \sum_i \left[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \right]
$$

Proper scoring rule for probabilities. Do not use squared error on the probability.

**Odds ratio interpretation**

A coefficient $w_j$ is a change in log-odds. $e^{w_j}$ is the odds ratio for a one-unit increase in feature $j$ (holding others fixed).

---

## Support Vector Machines (SVM)

**Soft-margin primal objective**

$$
J(w,b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max\bigl(0,\ 1 - y_i (w \cdot x_i + b)\bigr)
$$

Maximizes margin ($\frac{1}{2}\|w\|^2$) while allowing some violations controlled by $C$. Large $C$ → hard margin / less tolerance for errors; small $C$ → wider margin, more regularized.

**Kernel trick (example: RBF)**

$$
K(x_i, x_j) = \exp\Bigl(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\Bigr)
$$

Computes similarity in a higher-dimensional space without explicitly building it. Enables nonlinear boundaries.

---

## Decision Trees

**Classification impurity**

- Gini: $1 - \sum_k p_k^2$
- Entropy: $-\sum_k p_k \log p_k$

Split that reduces impurity the most (Information Gain / Variance Reduction).

**Regression leaf**

Prediction = mean (or median) of the target values in that leaf.

**Variance reduction (regression split criterion)**

$$
\text{Var reduction} = \text{Var}(parent) - \sum \frac{n_{child}}{n_{parent}} \text{Var}(child)
$$

---

## Ensembles

**Random Forest**

Average of many trees, each trained on a bootstrap sample + random feature subset at each split. Reduces variance of a single high-variance tree.

**Gradient Boosting (high-level)**

Each new tree is fit to the negative gradient of the loss of the current ensemble (i.e., the residual errors). Small learning rate + many trees.

---

## Clustering (K-Means)

**Within-Cluster Sum of Squares (WCSS / inertia)**

$$
\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
$$

Objective minimized by K-means. Elbow method plots WCSS vs $K$ to choose the number of clusters.

---

## Naive Bayes

**Bayes theorem**

$$
P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}
$$

**Naive assumption + ranking**

$$
P(y \mid x) \propto P(y) \prod_j P(x_j \mid y)
$$

Features are treated as independent given the class. Only the ranking matters for classification.

---

## Gradient Descent (general)

$$
\theta := \theta - \alpha \nabla_\theta J(\theta)
$$

Used when no closed form exists (logistic, neural nets, large linear models). Learning rate $\alpha$ controls step size.

---

*These are the formulas that appear repeatedly in the notes and interview questions. See individual notes for longer derivations and assumptions.*
