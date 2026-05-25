import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree

    def _expand_features(self, X):
        n_samples, n_features = X.shape
        poly_features = [np.ones(n_samples)]  # bias term
        for d in range(1, self.degree + 1):
            for i in range(n_features):
                poly_features.append(X[:, i] ** d)
        return np.vstack(poly_features).T

    def fit(self, X, y):
        X_poly = self._expand_features(X)
        self.theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
        self.poly_features_ = X_poly.shape[1]

    def predict(self, X):
        X_poly = self._expand_features(X)
        return X_poly @ self.theta

