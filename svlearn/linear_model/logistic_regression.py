import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        x = np.insert(x, 0, 1, axis=1)
        weights = np.ones(x.shape[1])
        for _ in range(self.epochs):
            y_pred = self._sigmoid(np.dot(x, weights))
            weights = weights + self.lr * (np.dot((y - y_pred), x) / x.shape[0])
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict_prob(self, x):
        y = np.dot(x, self.coef_) + self.intercept_
        return self._sigmoid(y)

    def predict(self, x):
        y = self.predict_prob(x)
        return np.where(y >= 0.5, 1, 0)