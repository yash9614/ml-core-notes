# Ensembles

Source: AdaBoost, gradient boosting, and XGBoost PDFs and notebooks, `RandomForest+Regression.pdf`, `12.2-Out+of+bag+evaluation.pdf`.

## Bagging and random forests

Bagging fits the same model on bootstrap samples and averages them. Variance drops. Bias stays about the same.

A random forest is bagged trees plus a random subset of features at each split. The feature subset decorrelates the trees. If every tree picks the same first split, averaging does little.

Out-of-bag evaluation: each tree misses about a third of the rows. Score those rows with the trees that did not see them. A free estimate, not a substitute for a final holdout if you also tuned hyperparameters on OOB.

## Boosting

Boosting adds models sequentially. Each one targets what the current sum gets wrong.

- AdaBoost: increase the weight of misclassified rows. Next tree focuses there.
- Gradient boosting: fit the next tree to the negative gradient of the loss. For squared error, that gradient is the residual. For log loss, it is not.
- XGBoost: gradient boosting with a regularized tree, column subsampling, and a second-order step. Small learning rate and more trees is the usual pattern. Early stopping on a validation set.

Boosting can overfit. It just does it more slowly than one deep tree. Learning rate, tree depth, and the number of rounds are the three knobs.

## Interview questions

1. Forest vs one tree? Lower variance. Less readable. Still cannot extrapolate past the training range.
2. OOB vs cross-validation? OOB is almost free for a forest. It is optimistic if you used it to pick every hyperparameter and then quote it as the test score.
3. AdaBoost vs GBM? Reweight rows vs fit the gradient. GBM generalizes to any differentiable loss.
4. Why is XGBoost strong on tables? Regularized trees, missing-value handling, interactions, no scaling required. It is not magic on raw images or text.

## Kaggle

House Prices with a forest, then a boosted model. IEEE-CIS Fraud Detection once the metric and the imbalance are familiar.
