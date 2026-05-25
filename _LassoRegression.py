import numpy as np;

class LassoRegression:
    def __init__(self, alpha=0.01, lr=0.01, n_iter=1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter

    def _soft_threshold(self, rho, alpha):
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0

    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0

        for _ in range(self.n_iter):
            y_pred = X @ self.w + self.b
            error = y_pred - y

            dw = (1/n) * (X.T @ error)
            db = (1/n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            #L1 soft-thresholding
            self.w = np.array([self._soft_threshold(wi, self.lr * self.alpha) for wi in self.w])

    def predict(self, X):
        return X @ self.w + self.b
