import numpy as np
import pandas as pd
import pickle
import argparse

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

def standardize_with_params(X, mu, sigma):
    return (X - mu) / sigma

def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--metrics_output_path', required=True)
    parser.add_argument('--predictions_output_path', required=True)
    args = parser.parse_args()
    
    with open(args.model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    mu = model_data['mu']
    sigma = model_data['sigma']
    features = model_data['features']
    target_col = model_data['target_col']
    
    df = pd.read_csv(args.data_path)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
    
    df1 = df.dropna(subset=[target_col]).copy()
    for c in features:
        if df1[c].isna().any():
            df1[c] = df1[c].fillna(df1[c].median())
    
    X = df1[features].to_numpy().astype(float)
    y_true = df1[target_col].to_numpy().astype(float).reshape(-1, 1)
    
    Xs = standardize_with_params(X, mu, sigma)
    
    if hasattr(model, 'degree'):
        y_pred = model.predict(Xs)
    else:
        Xsi = add_intercept(Xs)
        y_pred = model.predict(Xsi)
    
    mse_val = mse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    
    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse_val:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}\n")
        f.write(f"R-squared (R^2) Score: {r2_val:.2f}\n")
    
    pred_df = pd.DataFrame(y_pred.ravel())
    pred_df.to_csv(args.predictions_output_path, index=False, header=False)
    
    print(f"Metrics saved to: {args.metrics_output_path}")
    print(f"Predictions saved to: {args.predictions_output_path}")
    print(f"MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.2f}")

if __name__ == "__main__":
    main()