import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import mlflow

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/air_medical_transport_demand.csv"
REPORT_PATH = "reports/log_ols_regression_summary.txt"


def main():
    print("Starting log OLS regression script...")

    df = pd.read_csv(DATA_PATH)

    target = "daily_transport_requests"

    features = [
        "region",
        "day_of_week",
        "month",
        "weekend_flag",
        "holiday_flag",
        "avg_temperature",
        "precipitation",
        "wind_speed",
        "visibility",
        "trauma_index",
        "hospital_transfer_volume",
        "rurality_index",
    ]

    # Log-transform dependent variable
    # log1p handles zeros safely: log(1 + y)
    df["log_daily_transport_requests"] = np.log1p(df[target])

    X = df[features]
    y = df["log_daily_transport_requests"]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=["region"], drop_first=True)

    # Force all columns numeric
    X = X.astype(float)

    # Add intercept
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test, original_y_train, original_y_test = train_test_split(
        X,
        y,
        df[target],
        test_size=0.2,
        random_state=42,
    )

    mlflow.set_experiment("air_medical_log_ols_regression")

    with mlflow.start_run():
        model = sm.OLS(y_train, X_train).fit()

        log_predictions = model.predict(X_test)

        # Convert log predictions back to original demand scale
        predictions = np.expm1(log_predictions)

        mae = mean_absolute_error(original_y_test, predictions)
        mse = mean_squared_error(original_y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(original_y_test, predictions)

        print("\nOLS Regression Summary:")
        print(model.summary())

        print("\nModel Performance on Original Demand Scale:")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²:   {r2:.3f}")

        print("\nP-values:")
        print(model.pvalues.sort_values())

        print("\nCoefficients:")
        print(model.params.sort_values())

        mlflow.log_param("model_type", "Log-Transformed OLS Regression")
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("mae_original_scale", mae)
        mlflow.log_metric("rmse_original_scale", rmse)
        mlflow.log_metric("r2_original_scale", r2)
        mlflow.log_metric("adj_r_squared_log_scale", model.rsquared_adj)
        mlflow.log_metric("aic", model.aic)
        mlflow.log_metric("bic", model.bic)

        os.makedirs("reports", exist_ok=True)

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(str(model.summary()))
            f.write("\n\nP-values:\n")
            f.write(str(model.pvalues.sort_values()))
            f.write("\n\nCoefficients:\n")
            f.write(str(model.params.sort_values()))
            f.write("\n\nModel Performance on Original Demand Scale:\n")
            f.write(f"MAE:  {mae:.3f}\n")
            f.write(f"RMSE: {rmse:.3f}\n")
            f.write(f"R²:   {r2:.3f}\n")

        mlflow.log_artifact(REPORT_PATH)

        print(f"\nRegression report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()