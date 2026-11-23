# K-Nearest Neighbors

Source: Krish Naik materials + core concepts.  
**Industrial relevance and practical patterns added.**

## What it is

Store the training rows. For a new point, find the k closest rows and vote (classification) or average them (regression).

No weight vector. The model is the data. Prediction cost grows with the dataset, which is why a KD-tree or ball tree exists.

## Distance and scale

Euclidean distance is dominated by the column with the largest range. **Always scale features.** A salary column will drown age if unscaled. Categorical features need a defined distance (one-hot + Euclidean is common but not perfect).

Weighted KNN: closer neighbors can vote with higher weight (inverse distance). Uniform weight treats all k neighbors equally.

## Choosing k

k = 1 memorizes the training data. Very large k predicts the global mean. Odd k avoids ties in binary classification. Pick k on a validation curve.

**Curse of dimensionality**: in high dimension every point is far from every other point. Distances stop meaning “similar.” KNN gets worse as you add junk features. Prefer PCA or feature selection before KNN.

## KD-tree and ball tree

- **KD-tree**: recursively split on one axis at a time. Fast in low dimension (roughly d ≤ 20). Degrades in high dimension.
- **Ball tree**: partitions into nested hyperspheres. Often better than KD-tree when dimension is higher.
- **Brute force**: compute every distance. Wins when n is small or d is large.

sklearn chooses automatically when `algorithm='auto'`.

### Practical code

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])

# Quick validation
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print("Mean accuracy:", scores.mean())

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## Industrial relevance

- Rarely the final production model on large tabular data (tree ensembles usually win). Useful as a baseline or when the decision boundary is highly local.
- Prediction latency grows with dataset size — problematic for real-time systems with millions of rows unless approximate nearest neighbor methods are used.
- Still common in recommendation systems (item-item or user-user collaborative filtering) and some anomaly detection setups.
- Always scale. Consider dimensionality reduction if features are high-dimensional.

## Interview questions

1. What is trained in KNN? Almost nothing — the model is the stored data.
2. Why must you scale features before KNN?
3. How do you choose k? What happens if k is too small or too large?
4. Explain the curse of dimensionality and how it affects KNN.
5. What is the difference between KD-tree, ball tree, and brute force?
6. When would you use weighted KNN instead of uniform?
7. Why is KNN rarely used as the primary model on large production tabular datasets?
8. How does KNN handle categorical features?
