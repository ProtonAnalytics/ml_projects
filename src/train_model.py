import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/raw/air_medical_transport_demand.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/air_medical_demand_model"


def main():
    print("Starting training script...")
    df = pd.read_csv(DATA_PATH)

    target = "daily_transport_requests"

    features = [
       # "region",
        "day_of_week",
        "month",
       #"weekend_flag",
      #  "holiday_flag",
        "avg_temperature",
      #  "precipitation",
      #  "wind_speed",
      #  "visibility",
       "trauma_index",
        "hospital_transfer_volume",
        #"rurality_index",
    ]

    X = df[features]
    y = df[target]

    categorical_features = ["region"]
    numeric_features = [col for col in features if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    mlflow.set_experiment("air_medical_demand_forecasting")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        # -----------------------------------------
        # FEATURE IMPORTANCE ANALYSIS
        # -----------------------------------------

        feature_names = (
            pipeline.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        importances = pipeline.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False,
        )

        print("\nFeature Importance Rankings:")
        print(importance_df.to_string(index=False))

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        os.makedirs(MODEL_DIR, exist_ok=True)

        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path=MODEL_PATH,
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        print("Baseline model training complete.")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²:   {r2:.3f}")
        print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()