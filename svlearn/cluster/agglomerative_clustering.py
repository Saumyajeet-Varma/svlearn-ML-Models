import numpy as np

class AgglomerativeClustering:

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
    
    def proximity_matrix(self, x):
        n = len(x)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist[i][j] = dist[j][i] = np.sqrt(np.sum((x[i] - x[j]) ** 2))
        return dist
    
    def fit_predict(self, x):
        n = len(x)
        clusters = {i: [i] for i in range(n)}
        dist = self.proximity_matrix(x)
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_a, merge_b = -1, -1
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    cluster_i = clusters[keys[i]]
                    cluster_j = clusters[keys[j]]
                    curr_dist = min(dist[p][q] for p in cluster_i for q in cluster_j)
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        merge_a = keys[i]
                        merge_b = keys[j]
            clusters[merge_a].extend(clusters[merge_b])
            del clusters[merge_b]
        labels = np.zeros(n, dtype=int)
        cluster_id = 0
        for key in clusters:
            for idx in clusters[key]:
                labels[idx] = cluster_id
            cluster_id += 1
        return labels