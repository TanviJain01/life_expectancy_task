best_model = PolynomialRegression(degree=3)
best_model.fit(X_train, y_train)


y_train_pred = best_model.predict(X_train)

train_predictions_df = pd.DataFrame(y_train_pred, columns=['Predicted Life Expectancy'])
csv_output_path = "results/train_prediction.csv"
os.makedirs("results", exist_ok=True)
train_predictions_df.to_csv(csv_output_path, index=False, header=False)


train_mse = mse(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)


metrics_output_path = "results/train_metrics.txt"
with open(metrics_output_path, "w") as f:
    f.write("Regression Metrics:\n")
    f.write(f"Mean Squared Error (MSE): {train_mse:.6f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {train_rmse:.6f}\n")
    f.write(f"R-squared (R²) Score: {train_r2:.6f}\n")