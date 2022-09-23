from sklearn import cluster
import numpy as np


def k_means_cluster(data, number_of_clusters):
    clusters = cluster.KMeans(n_clusters=number_of_clusters)
    clusters.fit(data)
    return clusters.cluster_centers_, clusters.fit_transform(data)


def calculate_center_distance(clusters):
    max_cluster_difference = np.zeros([clusters.shape[1]])

    for i in range(clusters.shape[1]):
        diff = np.abs(clusters[i][:] - clusters[:][:])
        max_cluster_difference[i] = np.max(diff)

    return max_cluster_difference
