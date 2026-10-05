# Interview bank

Short answers. Say the answer, then the failure mode. The notes have the longer version.

## Basics

1. Supervised vs unsupervised? Labels vs no labels. Clustering is unsupervised. KNN classification is supervised.
2. Bias vs variance? Bias is error from a too-simple model. Variance is error from a model that chases the training sample. High bias underfits. High variance overfits.
3. Accuracy on a 99% negative set? Useless. Report precision, recall, PR-AUC, or the cost of each error.

## Linear and logistic

4. OLS closed form? \(w = (X^\top X)^{-1} X^\top y\), when the Gram matrix is invertible.
5. Why squared loss? Differentiable, penalizes big errors, matches Gaussian noise. Absolute loss is more robust.
6. Logistic coefficient? A change in log-odds. Exp of the coefficient is an odds ratio.
7. Why sigmoid? Squash the linear score into a probability. The boundary is still linear.
8. Class weight or threshold? Threshold moves the operating point. Class weight changes training. Use both only if you can say which job each one does.

## KNN, Bayes, SVM

9. KNN training? There is none, apart from storing rows and maybe building a KD-tree or ball tree.
10. How do you pick k? Validation curve. Small k overfits. Large k underfits toward the global mean.
11. Naive Bayes assumption? Features are independent given the class. It is false and the model often still works for text.
12. Gaussian vs multinomial vs Bernoulli NB? Continuous features, counts, binary flags.
13. What is the support vector? A training point on the margin or inside it. Remove the others and the boundary does not move, in the hard-margin case.
14. Kernel trick? Compute similarity in a higher-dimensional space without building that space. RBF is the default when the boundary is curved.

## Trees and ensembles

15. Split criterion? Gini or entropy for classes. Variance for numbers.
16. Why forests? Average many high-variance trees. Bagging plus random feature subsets decorrelates them.
17. Out-of-bag? Rows left out of a tree's bootstrap sample. A free validation estimate.
18. AdaBoost vs gradient boosting? AdaBoost reweights misclassified rows. GBM fits the next tree to the residual, or to the gradient of the loss.
19. Why does XGBoost need a learning rate? Each tree is a step. A small rate plus more trees generalizes better than one huge tree.
20. Can a tree extrapolate? No.

## Unsupervised and reduction

21. K-means objective? Minimize within-cluster sum of squares. It assumes round clusters of similar size.
22. K-means vs hierarchical? K-means needs k and a centroid. Hierarchical gives a dendrogram and does not need k up front.
23. DBSCAN parameters? `eps` and `min_samples`. It can mark noise and find non-round clusters. It fails when density varies.
24. Isolation Forest? Anomalies are easier to isolate with random splits, so they have a shorter path.
25. PCA is not feature selection. It builds new axes from linear combinations. The first component is the direction of maximum variance.

## Validation

26. Why not a single train/test split? One split can be lucky. K-fold averages that. Stratify if classes are rare.
27. Leakage? Fitting the scaler, encoder, or feature selection on the full data before splitting. Fit them inside the training fold only.
28. ROC-AUC vs PR-AUC? ROC can look fine when positives are rare. PR-AUC tracks precision as recall changes, which is the plot you want for fraud or churn.
