import numpy as np

class SimpleLinearRegression:
    def fit(self, X, y):
        X = X.flatten()
        self.w = np.sum((X - X.mean()) * (y - y.mean())) / np.sum((X - X.mean())**2)
        self.b = y.mean() - self.w * X.mean()

    def predict(self, X):
        X = X.flatten()
        return self.w * X + self.b