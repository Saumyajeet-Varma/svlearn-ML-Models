import numpy as np

class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, x, y):
        x = np.insert(x, 0, 1, axis=1)
        weights = np.linalg.inv(np.dot(x.T, x)).dot(x.T).dot(y)
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, x):
        y_pred = np.dot(x, self.coef_) + self.intercept_
        return y_pred