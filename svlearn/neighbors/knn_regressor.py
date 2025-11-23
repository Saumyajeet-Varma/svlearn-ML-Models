import numpy as np

class KNeighborsRegressor:

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
        k_values = [self.y_train[i] for i in k_indices]
        mean_val = np.mean(k_values)
        return mean_val

    def predict(self, x):
        x = np.array(x)
        return np.array([self._predict_single(sample) for sample in x])