# Clustering

Source: `Kmeans+clustering.pdf`, `Hierarichal+Clustering.pdf`, `DBCAN.pdf`, `silhoutteclustering.pdf`, `kmeans+vss+heirarichal.pdf`, and the matching notebooks.

## K-means

Pick k centroids. Assign each point to the nearest centroid. Move each centroid to the mean of its points. Repeat.

Objective: within-cluster sum of squares. It wants round clusters of similar size. It cannot find a ring. It needs k. Results depend on the start; run it several times.

Scale first. A wide column owns the distance.

Choose k with the elbow of inertia, then check silhouette. Elbow is a hint. Silhouette is a score. Neither knows the business meaning of a segment.

## Hierarchical

Start with each point as a cluster and merge the closest pair, or start with one cluster and split. The dendrogram is the output. You cut it to get k.

Linkage matters. Ward minimizes variance. Complete uses the farthest pair. Single uses the nearest pair and chains. No centroid step. Cost is higher. You do not have to pick k before seeing the tree.

## DBSCAN

A point is a core point if at least `min_samples` points lie within `eps`. Clusters are connected core points. Everything else is border or noise.

It finds non-round shapes and marks outliers. It fails when clusters have different densities, because one `eps` cannot serve both. There is no k.

## Interview questions

1. K-means loss? Within-cluster sum of squares.
2. Why not use K-means on a ring? The mean of a ring is the hole. Every point is far from that mean.
3. DBSCAN vs K-means? Density vs centroids. DBSCAN has noise points and no k. K-means has k and no noise label.

## Kaggle

[Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). Run K-means and DBSCAN on the same two columns and write down where they disagree.
