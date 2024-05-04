# # clustering_helpers.py

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, DBSCAN
# # fuzzy clustering
# from sklearn.mixture import GaussianMixture
# import pandas as pd

# def load_data(path):
#     data = pd.read_csv(path)
#     num_data = data.select_dtypes(include=[np.number])
#     return num_data


# def kmeans_clustering(df, n_clusters=2):
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(df)
#     labels = kmeans.labels_
#     centers = kmeans.cluster_centers_
#     # visualize clusters
#     plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=50, cmap='viridis')
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)




# # def kmeans_clustering(X, n_clusters=2):
# #     kmeans = KMeans(n_clusters=n_clusters)
# #     kmeans.fit(X)
# #     labels = kmeans.labels_
# #     centers = kmeans.cluster_centers_
# #     # plot clusters
# #     plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
# #     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# # def dbscan_clustering(X, eps=0.5, min_samples=5):
# #     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# #     dbscan.fit(X)
# #     labels = dbscan.labels_
# #     return labels

# # def gmm_clustering(X, n_components=2):
# #     gmm = GaussianMixture(n_components=n_components)
# #     gmm.fit(X)
# #     labels = gmm.predict(X)
# #     centers = gmm.means_
# #     return labels, centers