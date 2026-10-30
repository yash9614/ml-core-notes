# Core ML Formulas

Highlighted key formulas from the notes, with short explanations. Focus on what the formula *does* and when it matters.

> Formulas use GitHub's ```math fences for reliable rendering.

## Linear Regression

```math
\hat{y} = w_0 + w_1 x_1 + \cdots + w_p x_p = w^\top x + b
```

Predicts a continuous value. Linear in the weights (you can still add polynomial features of $x$).

**Ordinary Least Squares (OLS) loss**

```math
J(w) = \sum_i (y_i - \hat{y}_i)^2 = \| y - Xw \|^2
```

Minimizes squared residuals. Differentiable, penalizes large errors more, matches Gaussian noise assumption.

**Closed-form solution** (when $X^\top X$ is invertible)

```math
w = (X^\top X)^{-1} X^\top y
```

Direct solution. Gradient descent is used when the matrix is large or singular.

**Gradient descent update (for linear regression)**

```math
w := w - \alpha \cdot \frac{1}{n} X^\top (Xw - y)
```

$\alpha$ is the learning rate. Too large → diverges; too small → slow.

**Polynomial regression (degree 3 example)**

```math
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3
```

Still linear in the weights after expanding features.

**Ridge (L2)**

```math
J(\beta) = \mathrm{MSE} + \alpha \sum_j \beta_j^2
```

Shrinks coefficients. Helps with multicollinearity.

**Lasso (L1)**

```math
J(\beta) = \mathrm{MSE} + \alpha \sum_j |\beta_j|
```

Can drive coefficients to exactly zero (feature selection).

---

## Logistic Regression

**Sigmoid (maps linear score to probability)**

```math
P(y=1 \mid x) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}
```

Output stays in $(0,1)$. Decision boundary is still linear: where $w \cdot x + b = 0$.

**Log loss (binary cross-entropy)**

```math
J = -\frac{1}{n} \sum_i \left[ y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \right]
```

Proper scoring rule for probabilities. Do not use squared error on the probability.

**Log-odds (logit)**

```math
\log\left(\frac{p}{1-p}\right) = w \cdot x + b
```

A coefficient is a change in log-odds. $e^{w_j}$ is an odds ratio.

**Softmax (multiclass)**

```math
P(y=k \mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
```

Natural extension of the sigmoid to more than two classes.

---

## Support Vector Machines (SVM)

**Soft-margin primal objective**

```math
J(w,b) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max\bigl(0,\ 1 - y_i (w \cdot x_i + b)\bigr)
```

Maximizes margin while allowing violations controlled by $C$. Large $C$ → less tolerance for errors.

**Kernel trick (example: RBF)**

```math
K(x_i, x_j) = \exp\Bigl(-\gamma \|x_i - x_j\|^2\Bigr)
```

Computes similarity in a higher-dimensional space without explicitly building it.

---

## Decision Trees

**Classification impurity**

- Gini: $1 - \sum_k p_k^2$
- Entropy: $-\sum_k p_k \log p_k$

Split that reduces impurity the most.

**Regression leaf**

Prediction = mean (or median) of the target values in that leaf.

**Variance reduction (regression split criterion)**

```math
\text{Var reduction} = \text{Var}(parent) - \sum \frac{n_{child}}{n_{parent}} \text{Var}(child)
```

---

## Ensembles

**Random Forest (classification)**

```math
\hat{y} = \mathrm{mode}\bigl(T_1(x), T_2(x), \ldots, T_B(x)\bigr)
```

**Random Forest (regression)**

```math
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
```

Average of many trees, each on a bootstrap sample + random feature subset. Reduces variance.

**Gradient Boosting (high-level)**

```math
F_m(x) = F_{m-1}(x) + \nu h_m(x)
```

Each new tree $h_m$ is fit to the negative gradient of the loss (residuals). $\nu$ is the learning rate.

**XGBoost objective (simplified)**

```math
\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)
```

where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ (regularization on number of leaves and leaf weights).

---

## Clustering (K-Means)

**Within-Cluster Sum of Squares (WCSS / inertia)**

```math
\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
```

Objective minimized by K-means. Elbow method plots WCSS vs $K$.

---

## Naive Bayes

**Bayes theorem**

```math
P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}
```

**Naive assumption + ranking**

```math
P(y \mid x) \propto P(y) \prod_j P(x_j \mid y)
```

Features are treated as independent given the class.

**Gaussian density (for continuous features)**

```math
P(x_i \mid y) = \frac{1}{\sqrt{2\pi \sigma_y^2}} \exp\Bigl(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\Bigr)
```

---

## Gradient Descent

**What a derivative measures**

```math
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
```

Instantaneous rate of change. The gradient $\nabla f$ is the vector of partial derivatives.

**General update rule**

```math
\theta := \theta - \alpha \nabla_\theta J(\theta)
```

Learning rate $\alpha$ controls step size.

**Example: mean squared error for one feature**

```math
\frac{\partial J}{\partial w} = -\frac{2}{n} \sum_i x_i (y_i - \hat{y}_i), \quad
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_i (y_i - \hat{y}_i)
```

---

## Evaluation Metrics

**MSE**

```math
\mathrm{MSE} = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2
```

**$R^2$**

```math
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
```

**Precision**

```math
\mathrm{Precision} = \frac{TP}{TP + FP}
```

**Recall**

```math
\mathrm{Recall} = \frac{TP}{TP + FN}
```

**F1**

```math
F1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
```

---

*These are the formulas that appear repeatedly in the notes and interview questions. See individual notes for longer derivations and assumptions.*
