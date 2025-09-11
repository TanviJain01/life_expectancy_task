import numpy as np
import pickle
import os
from data_preprocessing import preprocess_data

class LinearRegressionNormalEq:
    def __init__(self): 
        self.theta = None
    def fit(self, X, y):
        self.theta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        return self
    def predict(self, X):
        return X @ self.theta

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.theta = None

    def _poly_features(self, X):
        feats = [X]
        for d in range(2, self.degree+1):
            feats.append(X ** d)
        return np.hstack(feats)

    def fit(self, X, y):
        Xp = self._poly_features(X)
        Xpi = np.hstack([np.ones((Xp.shape[0], 1)), Xp])
        self.theta = np.linalg.pinv(Xpi.T @ Xpi) @ (Xpi.T @ y)
        self.X_shape = X.shape[1]
        return self

    def predict(self, X):
        Xp = self._poly_features(X)
        Xpi = np.hstack([np.ones((Xp.shape[0], 1)), Xp])
        return Xpi @ self.theta

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta = None

    def fit(self, X, y):
        n = X.shape[1]
        I = np.eye(n)
        I[0,0] = 0
        self.theta = np.linalg.pinv(X.T @ X + self.alpha * I) @ (X.T @ y)
        return self

    def predict(self, X):
        return X @ self.theta

def mse(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return float(np.mean((y_true - y_pred)**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot)

def train_models(data_path):
    X, y, mu, sigma, features, target_col = preprocess_data(data_path)
    
    models = [
        ("regression_model1.pkl", LinearRegressionNormalEq()),
        ("regression_model2.pkl", PolynomialRegression(degree=2)),
        ("regression_model3.pkl", RidgeRegression(alpha=1.0)),
        ("regression_model_final.pkl", PolynomialRegression(degree=3))
    ]
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    for model_name, model in models:
        if isinstance(model, PolynomialRegression):
            X_no_intercept = X[:, 1:]
            model.fit(X_no_intercept, y)
            y_pred = model.predict(X_no_intercept)
        else:
            model.fit(X, y)
            y_pred = model.predict(X)
        
        model_data = {
            'model': model,
            'mu': mu,
            'sigma': sigma,
            'features': features,
            'target_col': target_col
        }
        
        with open(f"models/{model_name}", "wb") as f:
            pickle.dump(model_data, f)
        
        if model_name == "regression_model_final.pkl":
            mse_val = mse(y, y_pred)
            rmse_val = rmse(y, y_pred)
            r2_val = r2_score(y, y_pred)
            
            with open("results/train_metrics.txt", "w") as f:
                f.write("Regression Metrics:\n")
                f.write(f"Mean Squared Error (MSE): {mse_val:.2f}\n")
                f.write(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}\n")
                f.write(f"R-squared (R^2) Score: {r2_val:.2f}\n")
            
            import pandas as pd
            pred_df = pd.DataFrame(y_pred.ravel())
            pred_df.to_csv("results/train_predictions.csv", index=False, header=False)

if __name__ == "__main__":
    train_models("data/train_data.csv")