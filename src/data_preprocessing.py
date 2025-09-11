import numpy as np
import pandas as pd

def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma

def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def preprocess_data(csv_path, target_col=None):
    df = pd.read_csv(csv_path)
    
    if target_col is None:
        possible_targets = [c for c in df.columns if "life" in c.lower() and "expect" in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            target_col = df.columns[-1]
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for c in num_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
    
    features = [c for c in num_cols if c != target_col]
    
    df1 = df.dropna(subset=[target_col]).copy()
    
    for c in features:
        if df1[c].isna().any():
            df1[c] = df1[c].fillna(df1[c].median())
    
    X = df1[features].to_numpy().astype(float)
    y = df1[target_col].to_numpy().astype(float).reshape(-1, 1)
    
    Xs, mu, sigma = standardize(X)
    Xsi = add_intercept(Xs)
    
    return Xsi, y, mu, sigma, features, target_col