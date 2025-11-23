import numpy as np
from collections import Counter

class KNeighborsClassifier:

    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def _euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, x):
        x = np.array(x)
        return np.array([self._predict_single(sample) for sample in x])