# Anomaly detection and PCA

Source: Isolation Forest and LOF PDFs, `Isolation+Anamoly+Detection.ipynb`, `PCA.pdf`, `Principal+Component+Analysis+(PCA)+Implementation.ipynb`.

## Isolation Forest

Build random trees by picking a random feature and a random split. An outlier is separated in few splits, so its average path is short. The anomaly score comes from that path length.

No density estimate. Works in higher dimension better than distance methods. `contamination` is your guess of the outlier rate, not a learned fact.

## Local Outlier Factor

Compare a point's density with the density of its neighbors. A point in a sparse region next to a tight cluster scores as an outlier. A point in a globally sparse but locally uniform region does not.

LOF is local. Isolation Forest is global and random. Use LOF on a small table with clusters of different density. It gets expensive in high dimension.

## PCA

Rotate the data onto axes that are uncorrelated and ordered by variance. Component 1 is the direction that spreads the points the most. Later components are orthogonal to the earlier ones.

Steps: center the columns, optionally scale, compute the covariance eigenvectors, project.

Use it to compress, to plot, or to feed a distance model. Do not use it as a feature-importance ranking. A component is a mixture of original columns. Dropping a low-variance component can drop a rare but predictive signal.

How many components: a cumulative variance cutoff such as 90 or 95 percent, checked against the downstream validation score.

## Interview questions

1. Isolation Forest intuition? Outliers are easier to isolate, so the path is shorter.
2. LOF vs a global distance cutoff? LOF compares local densities. A global cutoff fails when one cluster is tight and another is loose.
3. PCA vs feature selection? PCA mixes columns. Selection keeps or drops original columns.
4. Do you scale before PCA? Yes if units differ. No if the units are already the same and variance is the signal.

## Kaggle

[Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) for Isolation Forest against a supervised baseline. Digit Recognizer for PCA, then a classifier on the components.
