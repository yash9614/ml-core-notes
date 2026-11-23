# Unsupervised Learning – Key Takeaways

Source: codebasics.io Takeaways – Unsupervised Learning.  
**Industrial relevance and practical patterns added.**

No labels. Goal is to discover structure: clusters, density, outliers. In industry, unsupervised methods are most often used for customer segmentation, anomaly detection, and exploratory analysis rather than as the final production model.

## K-Means

1. Choose K centroids (randomly or with k-means++).
2. Assign every point to the nearest centroid.
3. Move each centroid to the mean of its points.
4. Repeat until assignments stop changing.

**Choosing K**:  
- Plot SSE (sum of squared distances to centroids) vs. K.  
- Look for the “elbow” where SSE stops dropping sharply.  
- In sklearn, `inertia_` is the SSE.  
- Also consider silhouette score and business interpretability.

### Practical elbow method

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("k")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method")
plt.show()
```

**Tips**:  
- Scale features first (min-max or standard).  
- Sensitive to initialization and outliers.  
- Works best when clusters are roughly spherical and of similar size.  
- Use k-means++ initialization for better results.

**Industrial relevance**:  
- Very common for customer segmentation.  
- Always validate clusters with business stakeholders—mathematical clusters may not be actionable.  
- Not suitable when clusters have different densities or non-spherical shapes.

## Hierarchical Clustering

Builds a tree of clusters.

**Agglomerative** (bottom-up, most common):  
1. Each point starts as its own cluster.  
2. Merge the two closest clusters.  
3. Repeat until one cluster remains.  
4. Visualize with a dendrogram.

**Linkage methods**:  
- Single: min distance between points  
- Complete: max distance  
- Average: mean distance  
- Ward: minimizes increase in within-cluster variance (often preferred)

**Industrial relevance**:  
- Useful for smaller datasets where you want a hierarchy (e.g., product taxonomy).  
- Computationally expensive for large data. Dendrograms are excellent for stakeholder communication.

## DBSCAN

Density-based. Parameters: `eps` (neighborhood radius) and `min_samples`.

- **Core point**: ≥ min_samples neighbors within eps  
- **Border point**: reachable from a core point but not itself a core  
- **Noise / outlier**: neither

**Advantages**:  
- Finds arbitrary shapes  
- Automatically detects outliers  
- Does not require specifying number of clusters  

**Industrial relevance**:  
- Strong choice for anomaly detection and when cluster shapes are unknown.  
- Sensitive to parameter choice; often needs domain knowledge.  
- Does not scale as easily as K-Means to very large datasets without approximations.

## Interview Questions

1. What is the difference between K-Means and hierarchical clustering?
2. How do you choose the optimal number of clusters in K-Means?
3. When would you use DBSCAN instead of K-Means?
4. What are the main limitations of K-Means?
5. Explain the role of linkage methods in hierarchical clustering.
6. How would you use unsupervised learning for customer segmentation in a real company? What would success look like?
7. How does DBSCAN identify outliers?
