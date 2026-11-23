# Unsupervised Learning – Key Takeaways

Source: codebasics.io Takeaways – Unsupervised Learning.

No labels. Goal is to discover structure: clusters, density, outliers.

## K-Means

1. Choose K centroids (randomly).
2. Assign every point to the nearest centroid.
3. Move each centroid to the mean of its points.
4. Repeat until assignments stop changing.

**Choosing K**:  
- Plot SSE (sum of squared distances to centroids) vs. K.  
- Look for the “elbow” where SSE stops dropping sharply.  
- In sklearn, `inertia_` is the SSE.

**Tips**:  
- Scale features first (min-max or standard).  
- Sensitive to initialization and outliers.  
- Works best when clusters are roughly spherical and of similar size.

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
- Ward: minimizes increase in within-cluster variance

SciPy provides the tools; sklearn has `AgglomerativeClustering`.

## DBSCAN

Density-based. Parameters: `eps` (neighborhood radius) and `min_samples`.

- **Core point**: ≥ min_samples neighbors within eps  
- **Border point**: reachable from a core point but not itself a core  
- **Noise / outlier**: neither

**Advantages**:  
- Finds arbitrary shapes  
- Automatically detects outliers  
- Does not require specifying number of clusters  

**Tuning**: trial-and-error + scatter plots. Excellent for outlier detection.
