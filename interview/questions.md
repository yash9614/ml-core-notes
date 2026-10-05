# Interview questions by module

Say the answer, then the failure mode. Notes have the longer version.

## 01 Landscape

Note: [notes/01-ml-landscape.md](../notes/01-ml-landscape.md)

1. Supervised vs unsupervised? Labels vs no labels. Clustering is unsupervised. KNN classification is supervised.
2. Instance-based vs model-based? KNN stores rows and compares. Linear regression stores a weight vector and drops the rows.
3. Parametric vs non-parametric? Parametric has a fixed number of weights. KNN and an unpruned tree grow with the data.
4. Classification vs clustering? Classification has known class names. Clustering invents groups.

## 02 Linear regression

Note: [notes/02-linear-regression.md](../notes/02-linear-regression.md)

1. What does linear mean? Linear in the weights, not "the raw plot is a straight line."
2. OLS closed form? $w = (X^\top X)^{-1} X^\top y$, when $X^\top X$ is invertible.
3. Why squared loss? Differentiable, penalizes big errors, matches Gaussian noise. Absolute loss is more robust and fits a median.
4. What does a coefficient mean? Change in $y$ for a one-unit change in that feature, holding the others fixed. Not a causal effect.
5. Multicollinearity? Predictions can still be good. Individual coefficients cannot be trusted.
6. How do you see a bad line? Residual plot curves or fans out.
7. $R^2$ trap? It rises when you add junk features. Use adjusted $R^2$ or a holdout.

## 03 Regularization

Note: [notes/03-regularization.md](../notes/03-regularization.md)

1. Ridge vs Lasso? Ridge shrinks weights. Lasso can set them to exactly zero.
2. When Elastic Net? Wide tables with correlated features. Lasso alone picks one of a correlated pair at random.
3. Why scale before Ridge or Lasso? The penalty is on the weight, and the weight depends on the feature's unit.
4. Is polynomial regression a nonlinear model? Nonlinear in $x$, linear in the weights.
5. What is `alpha` vs `C`? `alpha` is $\lambda$. `C` on logistic regression is $1/\lambda$.

## 04 Logistic regression

Note: [notes/04-logistic-regression.md](../notes/04-logistic-regression.md)

1. Why not linear regression on a yes/no label? Outputs leave $[0, 1]$, and outliers yank the line.
2. Why the sigmoid? It maps the score $w \cdot x$ to a probability. The boundary $w \cdot x = 0$ is still linear.
3. Loss? Log loss, not squared error on the probability.
4. What is a coefficient? A change in log-odds. $e^{w_j}$ is an odds ratio.
5. When is 0.5 the wrong threshold? Rare positives, or unequal error costs. Tune it on precision-recall.
6. One-vs-rest vs softmax? One binary model per class, or one joint model. Softmax is the multi-class sigmoid.
7. Why is sklearn logistic regression already regularized? Default penalty is L2. `C` is the inverse strength.

## 05 KNN

Note: [notes/05-knn.md](../notes/05-knn.md)

1. What is trained? Nothing, except storing rows and maybe a KD-tree or ball tree.
2. How do you pick $k$? Validation curve. Small $k$ overfits. Large $k$ collapses to the global mean.
3. Why scale? Euclidean distance is owned by the column with the largest range.
4. Why does KNN die in high dimension? Distances stop separating near from far.
5. KD-tree vs brute force? Trees skip most rows in low dimension. Brute force wins when dimension is high.

## 06 Naive Bayes

Note: [notes/06-naive-bayes.md](../notes/06-naive-bayes.md)

1. What is naive? Features are treated as independent given the class.
2. Write the ranking rule. $P(y \mid x) \propto P(y) \prod_j P(x_j \mid y)$.
3. Gaussian vs multinomial vs Bernoulli? Continuous columns, word counts, binary flags.
4. Why Laplace smoothing? An unseen word would otherwise make the whole product zero.
5. Why does it still work on text? Only the class ranking matters, and each word probability is a stable count.

## 07 SVM

Note: [notes/07-svm.md](../notes/07-svm.md)

1. What is the margin? The empty band around the boundary. SVM maximizes it.
2. What is a support vector? A training point on the margin or inside it. Delete a non-support point and the hard-margin boundary stays.
3. What does `C` do? Large `C` punishes violations and can overfit. Small `C` allows a wider margin.
4. Kernel trick? Similarity in a higher-dimensional space without building that space. RBF is the default curved kernel.
5. SVM vs logistic regression? Margin vs likelihood. Logistic gives probabilities more naturally.
6. What is the epsilon-tube in SVR? Errors inside it are ignored.

## 08 Decision trees

Note: [notes/08-decision-trees.md](../notes/08-decision-trees.md)

1. Split criterion? Gini $1 - \sum_k p_k^2$ or entropy $-\sum_k p_k \log p_k$ for classes. Variance for numbers.
2. Numeric split? Sort the column, try thresholds between adjacent values, keep the best impurity drop.
3. Regression leaf? The mean of the rows in that leaf. The prediction is a step, not a line.
4. Why is one tree unstable? The top split decides every split below it.
5. Can it extrapolate? No. It cannot predict past the highest leaf mean it saw.
6. How do you stop overfitting? `max_depth`, `min_samples_leaf`, or cost-complexity pruning.

## 09 Ensembles

Note: [notes/09-ensembles.md](../notes/09-ensembles.md)

1. Why a forest? Average many high-variance trees. Random feature subsets keep them from making the same first split.
2. Out-of-bag? Rows left out of a tree's bootstrap sample. A free validation estimate. Optimistic if you also tuned on it.
3. AdaBoost vs gradient boosting? AdaBoost reweights missed rows. GBM fits the next tree to the gradient of the loss.
4. Why a small learning rate in XGBoost? Each tree is a step. A small rate plus more trees beats one huge tree.
5. Can a forest extrapolate? No. Averaging trees that cannot extrapolate still cannot.

## 10 Clustering

Note: [notes/10-clustering.md](../notes/10-clustering.md)

1. K-means objective? Within-cluster sum of squares. It wants round clusters of similar size.
2. Why does K-means fail on a ring? The centroid falls in the hole.
3. How do you pick $k$? Elbow is a hint. Silhouette is a score. Neither knows the business meaning.
4. Hierarchical vs K-means? A dendrogram, no $k$ required up front, more expensive.
5. DBSCAN parameters? `eps` and `min_samples`. It can mark noise. It fails when cluster densities differ.

## 11 Anomaly detection and PCA

Note: [notes/11-anomaly-and-pca.md](../notes/11-anomaly-and-pca.md)

1. Isolation Forest? Outliers take fewer random splits to isolate, so the path is shorter.
2. LOF vs Isolation Forest? LOF compares local density. Isolation Forest is a global random partition.
3. What is `contamination`? Your guess of the outlier rate, not a learned number.
4. PCA in one line? New uncorrelated axes, ordered by variance. Component 1 spreads the points the most.
5. PCA vs feature selection? PCA mixes columns. Selection keeps or drops original columns.
6. Scale before PCA? Yes if units differ.

## 12 Metrics and validation

Note: [notes/12-metrics-and-validation.md](../notes/12-metrics-and-validation.md)

1. RMSE vs MAE? RMSE punishes large errors more.
2. When is accuracy useless? A 99% negative set. Report precision, recall, or PR-AUC.
3. Precision vs recall? Precision when a false alarm is expensive. Recall when a miss is expensive.
4. ROC-AUC vs PR-AUC? ROC can look fine on rare positives. PR-AUC is the fraud and churn plot.
5. Bias vs variance? Too simple vs chasing the sample. High bias underfits. High variance overfits.
6. Why K-fold? One split can be lucky. Stratify if a class is rare. Do not shuffle time series.
7. Leakage? Scaler, encoder, or target encoding fit on the full column before the split.
