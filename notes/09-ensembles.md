# Ensembles

Source: AdaBoost, gradient boosting, and XGBoost PDFs and notebooks, `RandomForest+Regression.pdf`, `12.2-Out+of+bag+evaluation.pdf`.

**Exercise notebook:** [ensemble_ex1_assignment.ipynb](09-ensembles/ensemble_ex1_assignment.ipynb)

## Bagging and random forests

Bagging fits the same model on bootstrap samples and averages them. Variance drops. Bias stays about the same.

A random forest is bagged trees plus a random subset of features at each split. The feature subset decorrelates the trees. If every tree picks the same first split, averaging does little.

**Out-of-bag evaluation**: each tree misses about a third of the rows. Score those rows with the trees that did not see them. A free estimate, not a substitute for a final holdout if you also tuned hyperparameters on OOB.

## Boosting

Boosting adds models sequentially. Each one targets what the current sum gets wrong.

### AdaBoost

Each training row starts with equal weight. After a tree is fit:

1. Compute the weighted error of that tree.
2. Give the tree a weight `alpha = 0.5 * log((1 - err) / err)`. Better trees get higher alpha.
3. Increase the weight of misclassified rows and decrease the weight of correctly classified rows.
4. Renormalize so the weights sum to 1.

The final prediction is a weighted vote of the trees. AdaBoost is sensitive to label noise because noisy points keep getting higher weight.

### Gradient boosting

Fit the next tree to the negative gradient of the loss. For squared error that gradient is the residual. For log loss it is not. The learning rate multiplies each tree’s contribution so that many small steps beat one large step.

### XGBoost

Gradient boosting with:

- A regularized tree objective (penalty on number of leaves and leaf weights).
- Second-order (Hessian) information for the split gain.
- Column subsampling.
- Missing-value handling (learns a default direction).

Small learning rate + more trees is the usual pattern. Early stopping on a validation set prevents overfit.

Boosting can overfit. It just does it more slowly than one deep tree. Learning rate, tree depth, and the number of rounds are the three main knobs.

## Interview questions

1. Forest vs one tree? Lower variance. Less readable. Still cannot extrapolate past the training range.
2. OOB vs cross-validation? OOB is almost free for a forest. It is optimistic if you used it to pick every hyperparameter and then quote it as the test score.
3. AdaBoost vs GBM? Reweight rows vs fit the gradient. GBM generalizes to any differentiable loss. AdaBoost is more sensitive to label noise.
4. Why is XGBoost strong on tables? Regularized trees, missing-value handling, interactions, no scaling required. It is not magic on raw images or text.
5. What is the AdaBoost tree weight? `0.5 * log((1 - err) / err)`. A tree with error 0.5 gets weight 0.

## Kaggle

House Prices with a forest, then a boosted model. IEEE-CIS Fraud Detection once the metric and the imbalance are familiar.
