
import mlflow
import pandas as pd

from pathlib import Path


# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"

REPORT_DIR = BASE_DIR / "reports"

MLFLOW_DB = BASE_DIR / "mlflow.db"


# ==========================================
# STEP 2: CONFIGURE MLFLOW WITH SQLITE
# ==========================================

tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment(
    "Smart_Grid_Intelligence"
)

print("MLflow SQLite database configured!")
print("Database:", MLFLOW_DB)


# ==========================================
# STEP 3: CLASSIFICATION MODEL FILES
# ==========================================

CLASSIFICATION_MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "Random Forest":
        "random_forest.pkl",

    "Gradient Boosting":
        "gradient_boosting.pkl",

    "XGBoost":
        "xgboost.pkl",

    "Isolation Forest":
        "isolation_forest.pkl"
}


# ==========================================
# STEP 4: REGRESSION MODEL FILES
# ==========================================

REGRESSION_MODEL_FILES = {

    "Ridge Regression":
        "ridge_regression_forecast.pkl",

    "Random Forest":
        "random_forest_forecast.pkl",

    "Gradient Boosting":
        "gradient_boosting_forecast.pkl",

    "XGBoost":
        "xgboost_forecast.pkl",

    "LightGBM":
        "lightgbm_forecast.pkl"
}


# ==========================================
# STEP 5: COMMON ARTIFACT LOGGING
# ==========================================

def log_model_artifact(model_name, model_files):

    filename = model_files.get(model_name)

    if filename is None:
        return

    model_path = ARTIFACT_DIR / filename

    if model_path.exists():

        mlflow.log_artifact(
            str(model_path),
            artifact_path="trained_model"
        )

        print("Model artifact logged:", filename)

    else:

        print("Warning: Model file not found:", model_path)


# ==========================================
# STEP 6: TRACK CLASSIFICATION EXPERIMENTS
# ==========================================

def track_classification():

    print("\n================================")
    print("CLASSIFICATION EXPERIMENTS")
    print("================================")

    file_path = (
        ARTIFACT_DIR /
        "anomaly_model_comparison.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            "Anomaly comparison CSV not found. "
            "Run train_anomaly.py first."
        )

    df = pd.read_csv(file_path)

    for _, row in df.iterrows():

        model_name = row["Model"]

        print("\nLogging:", model_name)

        with mlflow.start_run(
            run_name=f"Anomaly_{model_name}"
        ):

            # Experiment tags
            mlflow.set_tags({

                "task": "Anomaly Detection",
                "model": model_name,
                "project": "Smart Grid Intelligence"

            })

            # Parameters
            mlflow.log_params({

                "task_type": "classification",
                "model_name": model_name,
                "dataset": "Household Power Consumption"

            })

            # Evaluation metrics
            metrics = [

                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC"

            ]

            for metric in metrics:

                if metric in df.columns and pd.notna(row[metric]):

                    mlflow.log_metric(
                        metric.lower(),
                        float(row[metric])
                    )

            # Log trained model artifact
            log_model_artifact(
                model_name,
                CLASSIFICATION_MODEL_FILES
            )

            # Log comparison CSV
            mlflow.log_artifact(
                str(file_path),
                artifact_path="reports"
            )

    print("\nClassification tracking completed!")


# ==========================================
# STEP 7: TRACK REGRESSION EXPERIMENTS
# ==========================================

def track_regression():

    print("\n================================")
    print("REGRESSION EXPERIMENTS")
    print("================================")

    file_path = (
        ARTIFACT_DIR /
        "forecast_model_comparison.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            "Forecast comparison CSV not found. "
            "Run train_forecast.py first."
        )

    df = pd.read_csv(file_path)

    for _, row in df.iterrows():

        model_name = row["Model"]

        print("\nLogging:", model_name)

        with mlflow.start_run(
            run_name=f"Forecast_{model_name}"
        ):

            # Experiment tags
            mlflow.set_tags({

                "task": "Electricity Forecasting",
                "model": model_name,
                "project": "Smart Grid Intelligence"

            })

            # Parameters
            mlflow.log_params({

                "task_type": "regression",
                "model_name": model_name,
                "dataset": "Household Power Consumption"

            })

            # Evaluation metrics
            metrics = [

                "MAE",
                "RMSE",
                "R2_Score",
                "MAPE_Percent"

            ]

            for metric in metrics:

                if metric in df.columns and pd.notna(row[metric]):

                    mlflow.log_metric(
                        metric.lower(),
                        float(row[metric])
                    )

            # Log trained model artifact
            log_model_artifact(
                model_name,
                REGRESSION_MODEL_FILES
            )

            # Log comparison CSV
            mlflow.log_artifact(
                str(file_path),
                artifact_path="reports"
            )

    print("\nRegression tracking completed!")


# ==========================================
# STEP 8: TRACK PROJECT REPORTS
# ==========================================

def track_reports():

    print("\n================================")
    print("PROJECT REPORT TRACKING")
    print("================================")

    report_files = [

        "classification_comparison.png",

        "regression_comparison.png",

        "final_model_summary.csv"

    ]

    for filename in report_files:

        file_path = REPORT_DIR / filename

        if file_path.exists():

            with mlflow.start_run(
                run_name=f"Report_{filename}"
            ):

                mlflow.set_tag(
                    "task",
                    "Project Reports"
                )

                mlflow.log_artifact(
                    str(file_path),
                    artifact_path="reports"
                )

            print("Report logged:", filename)

        else:

            print("Report not found:", filename)


# ==========================================
# STEP 9: MAIN EXECUTION
# ==========================================

def main():

    print("\n========================================")
    print("SMART GRID INTELLIGENCE - MLFLOW")
    print("========================================")

    # Track classification models
    track_classification()

    # Track regression models
    track_regression()

    # Track project reports
    track_reports()

    print("\n========================================")
    print("MLFLOW TRACKING COMPLETED SUCCESSFULLY!")
    print("========================================")

    print("\nExperiment Name:")
    print("Smart_Grid_Intelligence")

    print("\nTracking URI:")
    print(tracking_uri)

    print("\nDatabase Location:")
    print(MLFLOW_DB)


if __name__ == "__main__":

    main()
