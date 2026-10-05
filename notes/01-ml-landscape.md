# ML landscape

Source: `2-Typesof+ML+technqiues.pdf`, `instance+based+vs+model+absed+learning.pdf`.

## The three settings

- Supervised: each row has a label. Predict a number (regression) or a class (classification).
- Unsupervised: no label. Find groups, density, or a compressed representation.
- Reinforcement: an agent acts, gets a reward, and updates a policy. Not in this ML folder.

Semi-supervised and self-supervised sit between the first two. A few labeled rows plus a lot of unlabeled rows, or a pretext task such as predicting a missing piece.

## Instance-based vs model-based

Instance-based methods store the training rows and answer a new point by comparing it to them. KNN is the clean example. There is almost no training step. Prediction cost grows with the dataset.

Model-based methods fit parameters, then throw away the individual rows. Linear regression stores a weight vector. A decision tree stores splits. Prediction does not scan the training set.

## Parametric vs non-parametric

Parametric: a fixed number of parameters, regardless of how many rows you add. Linear and logistic regression.

Non-parametric: capacity grows with the data. KNN, decision trees before you prune them, kernel SVM.

## Batch vs online

Batch fits on the whole training set. Online or mini-batch updates from a stream. Linear models and neural nets can be online. A freshly built decision tree usually is not.

## Interview questions

1. Is KNN supervised or unsupervised? Supervised if you use labels. The neighbor search itself does not train a parameter vector.
2. Why is linear regression model-based and KNN instance-based? One stores weights. The other stores examples.
3. Give a problem that is not supervised. Customer segments with no churn label. That is clustering.
4. What is the difference between classification and clustering? Classification has known class names. Clustering invents groups.
