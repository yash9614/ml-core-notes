# Clustering

Source: codebasics unsupervised exercises + core concepts.  
**Industrial relevance and intuition added.**

## K-means

Pick k centroids. Assign each point to the nearest centroid. Move each centroid to the mean of its points. Repeat until assignments stabilize.

**Objective**: within-cluster sum of squares (inertia). It wants compact, roughly spherical clusters of similar size.

**Intuition**: Imagine dropping k pins on a map and repeatedly moving each pin to the center of the points closest to it. It works well when clusters are round and similarly sized; it fails on rings, elongated clusters, or clusters of very different densities.

**Practical points**:
- Scale features first — Euclidean distance is dominated by the feature with the largest range.
- Results depend on initialization; use `k-means++` or run multiple times.
- Choose k with the elbow of inertia, then check silhouette score. Neither knows the business meaning of a segment — always validate with stakeholders.

### Practical code sketch

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Final model
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
print("Silhouette:", silhouette_score(X_scaled, labels))
```

## Hierarchical

Start with each point as a cluster and merge the closest pair (agglomerative), or start with one cluster and split (divisive). The dendrogram is the output. You cut it to get k.

**Linkage**:
- Ward: minimizes variance (often preferred)
- Complete: farthest pair
- Single: nearest pair (can chain)
- Average: mean distance

No centroid step. More expensive than K-means. Useful when you want a hierarchy (e.g., product taxonomy) and the dataset is not huge.

## DBSCAN

A point is a **core point** if at least `min_samples` points lie within `eps`. Clusters are connected core points. Everything else is border or noise.

**Intuition**: It finds dense regions and labels sparse points as outliers. It can discover arbitrary shapes and does not require specifying k.

**Limitations**: Fails when clusters have different densities (one `eps` cannot serve both). Sensitive to parameter choice; often needs domain knowledge or visual inspection.

## Industrial relevance

- Clustering is most often used for **customer segmentation**, anomaly detection, and exploratory analysis — not as the final production model.
- Always validate clusters with business stakeholders. Mathematical clusters may not be actionable.
- Scale features. Prefer K-means++ initialization.
- For production anomaly detection, Isolation Forest or DBSCAN are common; K-means is rarely the best choice when outliers are the goal.
- Dendrograms are excellent for communicating hierarchical structure to non-technical stakeholders.

## Interview questions

1. What is the objective of K-means? Within-cluster sum of squares.
2. Why does K-means fail on a ring-shaped cluster? The centroid falls in the hole.
3. How do you choose the number of clusters? Elbow + silhouette + business interpretability.
4. What are the main limitations of K-means? Assumes spherical clusters of similar size, sensitive to initialization and outliers, requires k.
5. How does DBSCAN differ from K-means? Density-based, finds arbitrary shapes, automatically detects outliers, no need to specify k.
6. When would you use hierarchical clustering instead of K-means?
7. Why must you scale features before clustering?
8. How would you evaluate the quality of clusters in a real business setting?
9. What is the role of linkage methods in hierarchical clustering?
