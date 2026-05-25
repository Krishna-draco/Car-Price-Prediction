import numpy as np;

class RidgeRegression:
    def __init__(self, alpha=0.5, lr=0.01, n_iter=1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for _ in range(self.n_iter):
            y_pred = X @ self.w + self.b
            dw = (-2/n) * (X.T @ (y - y_pred)) + 2 * self.alpha * self.w
            db = (-2/n) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b