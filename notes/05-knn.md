# K-nearest neighbors

Source: `1,0-KNN+Classification+And+Regression.pdf`, `2.0-KD+Tree+Ball+Tree+.pdf`, `3.0-KNNClassifier.ipynb`, `3.0-KNNRegressor.ipynb`.

## What it is

Store the training rows. For a new point, find the k closest rows and vote (classification) or average them (regression).

No weight vector. The model is the data. Prediction cost grows with the dataset, which is why a KD-tree or ball tree exists.

## Distance and scale

Euclidean distance is dominated by the column with the largest range. Scale first, or a salary column drowns age. Categorical features need a defined distance. One-hot then Euclidean is a choice, not a law.

Weighted KNN: closer neighbors can vote with higher weight (inverse distance). Uniform weight treats all k neighbors equally.

## Choosing k

k = 1 memorizes. k very large predicts the global mean. Odd k avoids ties in binary classification. Pick k on a validation curve.

Curse of dimensionality: in high dimension every point is far from every other point. Distances stop meaning “similar.” KNN gets worse as you add junk features. PCA or feature selection comes before KNN, not after.

## KD-tree and ball tree

**KD-tree**: recursively split the space on one axis at a time (median of that feature). A query walks the tree and prunes branches that cannot contain a closer neighbor. Fast in low dimension (roughly d ≤ 20). Degrades to near-linear scan when dimension is high.

**Ball tree**: partitions into nested hyperspheres. Better than KD-tree when dimension is higher, because spheres adapt to the data distribution. Still loses to brute force in very high dimension.

**Brute force**: compute every distance. Wins when n is small or d is large, or when you need exact results and the tree overhead is not worth it.

sklearn chooses automatically based on the data shape when `algorithm='auto'`.

## Interview questions

1. Training complexity? Almost none, apart from storing rows and maybe building a tree. The cost is at prediction.
2. Why scale? Distance is unit-dependent.
3. KD-tree vs ball tree? Both partition space so a query does not scan every row. KD-tree splits on axes; ball tree uses spheres. Both degrade in high dimension. Brute force wins there.
4. Weighted vs uniform neighbors? Weighted gives more influence to closer points. Can help when density varies.

## Kaggle

[Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer). Scale pixels, try k from 1 to 15, and compare with a linear model.
