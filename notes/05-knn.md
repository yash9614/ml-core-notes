# K-nearest neighbors

Source: `1,0-KNN+Classification+And+Regression.pdf`, `2.0-KD+Tree+Ball+Tree+.pdf`, `3.0-KNNClassifier.ipynb`, `3.0-KNNRegressor.ipynb`.

## What it is

Store the training rows. For a new point, find the k closest rows and vote (classification) or average them (regression).

No weight vector. The model is the data. That is why prediction is slow and memory-heavy, and why a KD-tree or ball tree exists: they avoid a full scan when the dimension is not too high.

## Distance and scale

Euclidean distance is dominated by the column with the largest range. Scale first, or a salary column drowns age. Categorical features need a defined distance. One-hot then Euclidean is a choice, not a law.

## Choosing k

k = 1 memorizes. k very large predicts the global mean. Odd k avoids ties in binary classification. Pick k on a validation curve.

Curse of dimensionality: in high dimension every point is far from every other point. Distances stop meaning similar. KNN gets worse as you add junk features. PCA or feature selection comes before KNN, not after.

## Interview questions

1. Training complexity? Almost none, apart from storing rows and maybe building a tree. The cost is at prediction.
2. Why scale? Distance is unit-dependent.
3. KD-tree vs ball tree? Both partition space so a query does not scan every row. They degrade in high dimension. Brute force wins there.

## Kaggle

[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer). Scale pixels, try k from 1 to 15, and compare with a linear model.
