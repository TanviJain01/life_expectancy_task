import numpy as np
import pandas as pd

def basic_numeric_preprocess(df: pd.DataFrame, target_col: str):
    """
    Returns Xsi (standardized numeric features with intercept), y, feature_names, and (mu, sigma).
    - Drops rows with missing target
    - Keeps only numeric features (except target)
    - Median-imputes missing values in features
    - Standardizes features
    - Adds intercept column
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c != target_col]

    df = df.dropna(subset=[target_col]).copy()

    for c in features:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    X = df[features].to_numpy().astype(float)
    y = df[target_col].to_numpy().astype(float).reshape(-1, 1)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    Xsi = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
    feature_names = ["intercept"] + features
    return Xsi, y, feature_names, (mu, sigma)
