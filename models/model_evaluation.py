
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 2: LOAD CLASSIFICATION RESULTS
# ==========================================

def evaluate_classification():

    print("\nCLASSIFICATION MODEL EVALUATION")

    file_path = (
        ARTIFACT_DIR / "anomaly_model_comparison.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Run train_anomaly.py first."
        )

    df = pd.read_csv(file_path)

    print("\nAnomaly Detection Results:")
    print(df.to_string(index=False))

    # Best model based on F1 score
    best_model = df.loc[df["F1_Score"].idxmax()]

    print("\nBest Classification Model:")
    print(best_model["Model"])

    print("Best F1 Score:", best_model["F1_Score"])

    # Save classification comparison chart
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="Model",
        y="F1_Score"
    )

    plt.title("Anomaly Detection - Model Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("F1 Score")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "classification_comparison.png",
        dpi=300
    )

    plt.show()

    return df, best_model


# ==========================================
# STEP 3: LOAD REGRESSION RESULTS
# ==========================================

def evaluate_regression():

    print("\nREGRESSION MODEL EVALUATION")

    file_path = (
        ARTIFACT_DIR / "forecast_model_comparison.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Run train_forecast.py first."
        )

    df = pd.read_csv(file_path)

    print("\nElectricity Forecasting Results:")
    print(df.to_string(index=False))

    # Best model based on MAE
    # Exclude baseline from the ML model selection
    ml_models = df[
        df["Model"] != "Naive Baseline"
    ]

    best_model = ml_models.loc[
        ml_models["MAE"].idxmin()
    ]

    print("\nBest Regression Model:")
    print(best_model["Model"])

    print("Best MAE:", best_model["MAE"])

    # Save regression comparison chart
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="Model",
        y="MAE"
    )

    plt.title("Electricity Forecasting - Model Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Mean Absolute Error (MAE)")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "regression_comparison.png",
        dpi=300
    )

    plt.show()

    return df, best_model


# ==========================================
# STEP 4: SAVE FINAL MODEL COMPARISON
# ==========================================

def save_final_summary(
    classification_best,
    regression_best
):

    summary = pd.DataFrame([

        {
            "Task": "Anomaly Detection",
            "Best Model": classification_best["Model"],
            "Primary Metric": "F1 Score",
            "Metric Value": classification_best["F1_Score"]
        },

        {
            "Task": "Electricity Forecasting",
            "Best Model": regression_best["Model"],
            "Primary Metric": "MAE",
            "Metric Value": regression_best["MAE"]
        }

    ])

    output_file = REPORT_DIR / "final_model_summary.csv"

    summary.to_csv(
        output_file,
        index=False
    )

    print("\nFINAL MODEL SUMMARY")

    print(summary.to_string(index=False))

    print("\nSummary saved:", output_file)


# ==========================================
# STEP 5: MAIN EXECUTION
# ==========================================

def main():

    classification_df, classification_best = (
        evaluate_classification()
    )

    regression_df, regression_best = (
        evaluate_regression()
    )

    save_final_summary(
        classification_best,
        regression_best
    )

    print("\n================================")
    print("MODEL EVALUATION COMPLETED!")
    print("================================")


if __name__ == "__main__":

    main()
