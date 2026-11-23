# Anomaly Detection and PCA

Source: Krish Naik materials (Isolation Forest notebook) + core concepts.  
**Industrial relevance and practical patterns added.**

## Isolation Forest – Intuition

Build random trees by picking a random feature and a random split value between the min and max of that feature in the current node. An outlier is separated in few splits, so its average path is short. The anomaly score is a normalized function of that path length.

No density estimate. Works better in higher dimensions than pure distance methods. `contamination` is your guess of the outlier rate — it only sets the decision threshold, not the trees themselves.

### Practical code

```python
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

# Assume df is a DataFrame of features
clf = IsolationForest(contamination=0.05, random_state=42)
predictions = clf.fit_predict(df)   # -1 = outlier, 1 = inlier

outliers = df[predictions == -1]
print(f"Detected {len(outliers)} outliers")

# Simple visualization if 2D
if df.shape[1] == 2:
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=predictions, cmap="coolwarm")
    plt.title("Isolation Forest Anomalies")
    plt.show()
```

## Local Outlier Factor (LOF)

Compare a point’s local density with the average local density of its neighbors.

- Reachability distance: max(k-distance of the neighbor, actual distance).
- Local reachability density: inverse of the average reachability distance.
- LOF score ≈ 1 means the point is similar to its neighbors; score ≫ 1 means it is in a sparser region.

LOF is local. Isolation Forest is global and random. Use LOF on smaller tables with clusters of different density. It gets expensive in high dimension.

## PCA – Intuition

Rotate the data onto axes that are uncorrelated and ordered by variance. Component 1 is the direction that spreads the points the most. Later components are orthogonal to the earlier ones.

**Steps**: center the columns, optionally scale, compute covariance eigenvectors (or SVD), project.

Use it to compress, to plot, or to feed a distance-based model. Do not treat components as feature importance — a component is a mixture of original columns. Dropping a low-variance component can drop a rare but predictive signal.

How many components: cumulative variance cutoff (90–95%) checked against downstream validation performance.

## Industrial relevance

- Isolation Forest and LOF are common unsupervised baselines for anomaly detection when labeled outliers are scarce (fraud, sensor faults, network intrusion).
- In production, combine unsupervised anomaly scores with business rules or a small amount of labeled data for better precision.
- `contamination` should be set based on domain knowledge or estimated outlier rate; treat it as a hyperparameter.
- PCA is widely used for dimensionality reduction before distance-based methods (KNN, clustering) and for visualization. Always scale when features have different units.
- Monitor anomaly rates over time — a sudden jump can indicate data drift or a real incident.

## Interview questions

1. Explain the intuition behind Isolation Forest. Why are outliers easier to isolate?
2. What does the `contamination` parameter do? Does it change the trees?
3. When would you prefer LOF over Isolation Forest?
4. How does PCA differ from feature selection?
5. Why should you scale features before PCA?
6. How would you evaluate an unsupervised anomaly detection system in production?
7. What are the limitations of Isolation Forest on very high-dimensional or sparse data?
