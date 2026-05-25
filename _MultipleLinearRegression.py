import numpy as np

class MultipleLinearRegression:
    def fit(self, X, y):
        # Add bias (intercept) term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Use pseudo-inverse for numerical stability
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta