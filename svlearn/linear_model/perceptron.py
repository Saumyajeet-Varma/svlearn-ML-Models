import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.1, activation="step", epochs=100):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.activation_func = None

    def _relu(z):
        return z if z > 0 else 0
    
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _step(z):
        return 1 if z > 0 else 0
    
    def _tanh(z):
        return np.tanh(z)
    
    def _set_activation_func(self):
        return {"relu": self._relu, "sigmoid": self._sigmoid, "step": self._step, "tanh": self._tanh}[self.activation]

    def fit(self, x, y):
        n = len(x)
        self.weights = np.zeros(n)
        self.bias = 0
        self.activation_func = self._set_activation_func()
        y = np.where(y <= 0, 0, 1)
        for _ in range(self.epochs):
            for i, x_i in enumerate(x):
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(z)
                update = self.lr * (y[i] - y_pred)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        y = self.activation_func(z)
        return y